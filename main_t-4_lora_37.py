import sys
import os
import json
import warnings
import torch
from types import SimpleNamespace
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

import torch.nn as nn
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

import cv2
import torchvision.transforms as T
import random
from tqdm import tqdm
import gc
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import torch._dynamo
from peft import LoraConfig, get_peft_model

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm

import matplotlib.pyplot as plt
from torchvision import transforms as T


torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint"
)

warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad"
)


# --- 1. ENVIRONMENT & CACHE SETUP ---
os.environ["HF_HOME"] = "C:/MultiModal/hf_cache"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 2. THE DUMMY PATCH (To bypass transformers version issues) ---
import transformers.modeling_utils as modeling_utils
def dummy_prune_func(*args, **kwargs): pass
for attr in ['find_pruneable_heads_and_indices', 'prune_linear_layer', 'apply_chunking_to_forward']:
    if not hasattr(modeling_utils, attr):
        setattr(modeling_utils, attr, dummy_prune_func)

from transformers import (
    BitsAndBytesConfig, AutoModelForCausalLM, 
    SiglipImageProcessor, SiglipVisionModel
)

cv2.setNumThreads(0)

# Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

#################### Define the Multimodal Bridge
class MultimodalBridge(nn.Module):
    def __init__(self, siglip_dim=1152, egovlp_dim=768, llm_dim=3584, use_motion=True):
        super().__init__()
        self.use_motion = use_motion
        
        # Spatial Projection
        self.siglip_proj = nn.Sequential(
            nn.Linear(siglip_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
        if self.use_motion:
            # --- Motion projection (4 layers, matched depth to siglip_proj) ---
            # ego_embeds[:, :8, :] gives us CLS (index 0) + 7 early patch tokens.
            # This is parameter-free, stable from step 1, and richer than CLS alone.
            # The removed ego_compress (Linear 785->8) was an undertrained bottleneck
            # that needed the full dataset to converge — this slice needs nothing.
            self.egovlp_proj = nn.Sequential(
                nn.Linear(egovlp_dim, llm_dim),
                nn.LayerNorm(llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim)
            )
            # Learnable type embedding: marks the 8 motion tokens as categorically
            # different from the 729 spatial patch tokens before they enter the resampler.
            self.motion_type_embed = nn.Parameter(torch.zeros(1, 1, llm_dim))

    def forward(self, sig_embeds, ego_embeds=None):
        # --- FIX 1: float32 projection to prevent rank-1 collapse ---
        # Root cause: the 1152->3584 MLP expansion under bfloat16 (the LLM's compute
        # dtype when loaded with 4-bit NF4) causes LayerNorm saturation — all 729
        # SigLIP tokens collapse to the same direction regardless of input content.
        # This was confirmed by PCA showing PC2=0.0% across every sample.
        #
        # The fix: temporarily cast the projection module AND the input to float32,
        # run the forward pass, then cast the output back to the original dtype.
        # Casting only the input (sig_embeds.float()) is not enough — the Linear
        # weight is also bfloat16, causing the "mat1 and mat2 must have the same
        # dtype" RuntimeError. We must cast both together.
        #
        # No re-extraction of cached features needed — the .pt files stay as-is.
        # Peak extra VRAM: 729 tokens x 3584 dims x 4 bytes x 4 timesteps = ~42MB.
        orig_dtype = sig_embeds.dtype
        vis_tokens = self.siglip_proj.float()(sig_embeds.float()).to(orig_dtype)

        if self.use_motion and ego_embeds is not None:
            # Slice the first 8 EgoVLP tokens: index 0 = CLS (global motion summary),
            # indices 1-7 = early patch tokens carrying temporal structure.
            ego_sliced = ego_embeds[:, :8, :]                                         # [B, 8, 768]

            # Same float32 treatment for the motion projection for consistency.
            mot_tokens = self.egovlp_proj.float()(ego_sliced.float()).to(orig_dtype)  # [B, 8, 3584]

            # Add motion type embedding so the resampler can distinguish these
            # 8 tokens from the 729 spatial patch tokens
            mot_tokens = mot_tokens + self.motion_type_embed                          # [B, 8, 3584]

            # Concatenate: [B, 729, 3584] + [B, 8, 3584] -> [B, 737, 3584]
            return torch.cat([vis_tokens, mot_tokens], dim=1)

        return vis_tokens
    
class TemporalPerceiverResampler(nn.Module):
    def __init__(
        self,
        dim,
        num_latents=256,
        depth=2,
        num_heads=8,
        num_steps=4 # Added parameter for the temporal window size
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        
        # --- NEW: Temporal Positional Embeddings ---
        self.time_embed = nn.Parameter(torch.randn(num_steps, 1, dim)) # [4, 1, 3584]

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                # --- CHANGE 4: Self-attention on latents (standard Perceiver pattern) ---
                # Latents communicate with each other BEFORE reading from visual tokens.
                # This lets the 256 latents coordinate their queries, preventing them from
                # all attending to the same visual regions.
                "self_attn": nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    batch_first=True
                ),
                "norm0": nn.LayerNorm(dim),
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    batch_first=True
                ),
                "norm1": nn.LayerNorm(dim),

                "ff": nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            })
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, step_embeds_list):
        """
        step_embeds_list: A list of 4 tensors, each [B, 1514, D]
        """
        B = step_embeds_list[0].shape[0]
        
        # --- NEW: Add temporal embeddings BEFORE concatenation ---
        processed_steps = []
        for t_idx, step_embeds in enumerate(step_embeds_list):
            # Add the corresponding time embedding to every token in this step
            # time_embed[t_idx] is [1, D], which broadcasts across the 1514 sequence length
            processed_steps.append(step_embeds + self.time_embed[t_idx])
            
        # Concatenate the temporally-aware steps
        x = torch.cat(processed_steps, dim=1) # [B, 6056, D]

        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            # Self-attention with pre-norm (latents coordinate before reading visual tokens)
            self_attn_out, _ = layer["self_attn"](
                query=latents,
                key=latents,
                value=latents
            )
            latents = layer["norm0"](latents + self_attn_out)

            # Cross-attention with pre-norm via norm1 (was defined but never used before)
            # Pre-norm stabilises training: latents are normalised before attending to
            # the long visual sequence, preventing scale drift across depth=2 layers.
            cross_in = layer["norm1"](latents)
            attn_out, _ = layer["cross_attn"](
                query=cross_in,
                key=x,
                value=x
            )
            latents = latents + attn_out
            latents = latents + layer["ff"](latents)

        return self.norm(latents)
    
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim,
        num_latents=128,
        num_layers=2,
        num_heads=8
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    batch_first=True
                ),
                "ff": nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim*4),
                    nn.GELU(),
                    nn.Linear(dim*4, dim)
                )
            })
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, N, D]  (N = 1514 visual tokens)
        """

        B = x.shape[0]

        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        # [B, 128, D]

        for layer in self.layers:

            attn_out, _ = layer["cross_attn"](
                latents,
                x,
                x
            )

            latents = latents + attn_out
            latents = latents + layer["ff"](latents)

        return self.norm(latents)
    

class EpicKitchensDataset(Dataset):
    def __init__(self, csv_path, siglip_transform, egovlp_transform, history_len=4, tokenizer=None, cache_dir=None, dropout_prob=0.0):
        self.cache_dir = cache_dir
        df = pd.read_csv(csv_path)
        self.data = df.to_dict('records') 
        self.siglip_transform = siglip_transform
        self.egovlp_transform = egovlp_transform
        self.history_len = history_len
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer 
        self.dropout_prob = dropout_prob
        
        # We NO LONGER pre-compute prompts here.
        # It's fast enough to do it on the fly, and required for dynamic dropout.
    
    def __len__(self):
        return len(self.data) - self.history_len

    def __getitem__(self, idx):
        actual_idx = idx + self.history_len
        target_sample = self.data[actual_idx]
        target_text = target_sample['narration']
        
        # --- 1. Gather History Texts ---
        prev_actions = []
        for i in range(1, self.history_len + 1):
            prev_idx = actual_idx - i
            if prev_idx >= 0:
                prev_sample = self.data[prev_idx]
                if prev_sample["location"] == target_sample["location"]:
                    prev_actions.append(prev_sample["narration"])
                else:
                    prev_actions.append("[Start of Video]")
            else:
                prev_actions.append("[Start of Video]")
        prev_actions.reverse() # Now index 0 is t-4, index 3 is t-1

        # --- 2. Dynamic Prompt Generation & Dropout ---
        templates = [
            "Predict the next action.",
            "What action happens next?",
            "Next step in the video:",
            "What will the person do next?"
        ]
        instruction = random.choice(templates)
        
        # Apply Dropout Logic
        rng = random.Random(42 + idx) 
        
        # We manually build the exact string Qwen expects for its ChatML format
        # <|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n
        
        system_msg = (
            "You are an AI assistant that predicts the next action in egocentric cooking videos. "
            "Output ONLY a 2-3 word action phrase in the format '[verb] [noun]'. "
            "Examples of correct output: 'take knife', 'pour oil', 'close fridge', 'chop onion', 'put down plate'. "
            "Never chain multiple actions. Never explain. Output the phrase only."
        )
        
        print(f"self.dropout_prob {self.dropout_prob}")
        if self.dropout_prob > 0 and rng.random() < self.dropout_prob:
            # DROPOUT PATH: Masked prompt
            user_msg = f"Previous actions: None available. Predict based on video.\n\n{instruction}:"
        else:
            # STANDARD PATH: Full history prompt
            user_msg = f"Previous actions:\nt-4: {prev_actions[0]}\nt-3: {prev_actions[1]}\nt-2: {prev_actions[2]}\nt-1: {prev_actions[3]}\n\n{instruction}:"

        # Combine into the final raw text string
        prompt_text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

        # --- 3. Gather Visual History (Tensors) ---
        history_sig = []
        history_ego = []
        history_texts = []
        
        for i in range(self.history_len, 0, -1):
            prev_idx = actual_idx - i
            prev_sample = self.data[prev_idx]
            
            vid_id = os.path.basename(prev_sample["location"]).replace('.MP4', '')
            start_t = prev_sample["start_seconds"] 
            
            cache_file = os.path.join(self.cache_dir, f"{vid_id}_{start_t:.2f}.pt")
            
            if os.path.exists(cache_file):
                data = torch.load(cache_file)
                sig = data["sig_features"].squeeze(0)
                if data["ego_features"] is not None:
                    ego = data["ego_features"].squeeze(0)
                else:
                    ego = torch.zeros((785, 768)) 
            else:
                sig = torch.zeros((729, 1152))
                ego = torch.zeros((785, 768))
                
            history_sig.append(sig)
            history_ego.append(ego)
            history_texts.append(prev_sample["narration"])
            
        return {
            "history_sig": torch.stack(history_sig),  
            "history_ego": torch.stack(history_ego), 
            "history_texts": history_texts,          
            "prompt_text": prompt_text,
            "target_text": target_text,
            "video_info": {
                "video_path": target_sample.get("location", ""),
                "start_time": target_sample.get("start_seconds", 0),
                "end_time": target_sample.get("stop_seconds", 0) 
            }
        }
   
def collate_fn(batch):
    # Filter out failed loads
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
        
    return {
        # Shape becomes: [Batch, 4, 729, 1152]
        "history_sig": torch.stack([b["history_sig"] for b in batch]),
        
        # Shape becomes: [Batch, 4, 785, 768]
        "history_ego": torch.stack([b["history_ego"] for b in batch]),
        
        # Text lists
        "history_texts": [b["history_texts"] for b in batch],
        "prompt_text": [b["prompt_text"] for b in batch],
        "target_text": [b["target_text"] for b in batch],
        "video_info": [b["video_info"] for b in batch]
    }

class CachedFeatureDataset(Dataset):
    def __init__(self, feature_dir, annotations_df):
        self.feature_dir = feature_dir
        self.df = annotations_df

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = f"{row['video_id']}_{row['start_frame']}" # Or any unique ID
        
        # Load precomputed tensors
        sig_out = torch.load(os.path.join(self.feature_dir, f"{uid}_sig.pt"))
        ego_out = torch.load(os.path.join(self.feature_dir, f"{uid}_ego.pt"))
        
        return {
            "sig_out": sig_out, 
            "ego_out": ego_out,
            "prompt_text": row['prompt'],
            "target_text": row['target']
        }
    
def load_video_segment(video_path, start_sec, end_sec): 

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    siglip_frame_index = start_frame + (end_frame - start_frame) // 2
    ego_indices = np.linspace(start_frame, end_frame - 1, 4).astype(int)

    # Create a set of the exact 5 frames we care about for fast lookup
    target_indices = set(ego_indices)
    target_indices.add(siglip_frame_index)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_siglip = None
    frames_ego_dict = {} # Use a dict temporarily to keep them in order

    for i in range(start_frame, end_frame):
        # FAST: Just point to the next frame, don't decode the pixels yet
        ret = cap.grab() 
        if not ret:
            break

        # Only decode and convert if it's one of our 5 target frames
        if i in target_indices:
            ret, frame = cap.retrieve() # SLOW: Decode the pixels
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # SLOW: Color conversion

            if i == siglip_frame_index:
                frames_siglip = frame

            if i in ego_indices:
                frames_ego_dict[i] = frame

    cap.release()

    # Ensure ego frames are in the correct temporal order
    frames_ego = [frames_ego_dict[idx] for idx in ego_indices if idx in frames_ego_dict]

    return frames_siglip, frames_ego

def run_inference(hist_sig, hist_ego, prompt_text, top_k=3):
    bridge.eval()
    temporal_resampler.eval() # Make sure the resampler is in eval mode!
    
    tokens = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    prompt_ids = tokens.input_ids
    prompt_mask = tokens.attention_mask

    with torch.no_grad():
        # prompt_embeds = llm.model.embed_tokens(prompt_ids)
        prompt_embeds = llm.base_model.model.model.embed_tokens(prompt_ids)

        # --- 4-Step Temporal Window Logic ---
        all_visual_steps = []
        for t_idx in range(4): # Process t-4, t-3, t-2, t-1
            sig_t = hist_sig[:, t_idx, :, :]
            
            if USE_MOTION and hist_ego is not None:
                ego_t = hist_ego[:, t_idx, :, :]
                step_embeds = bridge(sig_t.to(llm.dtype), ego_t.to(llm.dtype))
            else:
                step_embeds = bridge(sig_t.to(llm.dtype), None)
            
            all_visual_steps.append(step_embeds)

        # Concatenate and Compress
        # visual_embeds_4step = torch.cat(all_visual_steps, dim=1)
        visual_embeds_4step = temporal_resampler(all_visual_steps)
        
        # Combine compressed visual history with the text prompt
        inputs_embeds = torch.cat([visual_embeds_4step, prompt_embeds], dim=1)
        
        # Create attention mask for generation (Visual tokens + Prompt tokens)
        visual_mask = torch.ones(visual_embeds_4step.shape[:2], dtype=prompt_mask.dtype, device="cuda")
        attention_mask = torch.cat([visual_mask, prompt_mask], dim=1)

        output_ids = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=8,          # Room to complete phrases; motion model was hitting 5-token ceiling
            do_sample=False,
            num_beams=top_k,
            num_return_sequences=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,    # Prevents "cut cut cut"
            length_penalty=-2.0,       # Doubled from -1.0: motion model over-generates, this pushes brevity harder
            no_repeat_ngram_size=3,    # Prevents "and take and take" loops
        )

        # DEBUG PRINTS
        # print(f"Full output shape: {output_ids.shape}")

        generated_texts = []
        for i in range(top_k):
            new_tokens = output_ids[i]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # --- CLEANING LOGIC ---
            # 1. Stop at Newlines (The model finished the action and is starting a new thought)
            text = text.split('\n')[0]
            
            # 2. Stop at "Human:" or "Assistant:" (Chat-format hallucinations)
            text = text.split('Human:')[0].split('Assistant:')[0]

            # 3. Truncate at first action-chaining connector.
            # The motion model tends to over-generate compound actions ("take knife and cut onion").
            # Targets are almost always a single "[verb] [noun]" phrase, so strip after the first connector.
            for connector in [' and ', ', and ', '. ', ',']:
                if connector in text:
                    text = text.split(connector)[0].strip()
                    break
            
            # 4. Clean up any trailing punctuation or extra spaces
            text = text.strip().rstrip('.,')
            
            generated_texts.append(text)

    return generated_texts

def load_egovlp(checkpoint_path):
    model = EgoModel(
        video_params=ego_config['video_params'],
        text_params=ego_config['text_params'],
        projection_dim=ego_config['projection_dim']
    )
    
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    return model.cuda().half().eval()


def set_seed(seed=42):
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
    
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set seed for environment
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Global seed set to: {seed}")

def get_unpooled_egovlp(motion_model, ego_input):
    unpooled_features = []
    def hook(module, input, output):
        unpooled_features.append(output)
        
    # Intercept output at the final LayerNorm before pooling
    handle = motion_model.video_model.norm.register_forward_hook(hook)
    _ = motion_model.video_model.forward_features(ego_input)
    handle.remove()
    
    return unpooled_features[0] # Returns [B, 785, 768]


def extract_and_save_features(dataloader, output_dir, vision_model, motion_model, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    vision_model.eval()
    
    # --- 1. Hook Setup ---
    hook_handle = None
    captured_features = []

    if motion_model:
        motion_model.eval()
        
        # This function captures the 3D tensor [B, 785, 768] during the forward pass
        def hook_fn(module, input, output):
            captured_features.append(output.detach().cpu())

        # Register hook on the final norm layer (the state before pooling)
        # This is the standard output layer for TimeSformer/EgoVLP backbones
        hook_handle = motion_model.video_model.norm.register_forward_hook(hook_fn)
        
    print(f"\nStarting Robust Feature Extraction (Hook Enabled) to '{output_dir}'...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            if batch is None:
                continue 
            
            # Process each item in the batch individually
            for i in range(len(batch["prompt_text"])):
                try:
                    v_info = batch["video_info"][i]
                    video_id = v_info['video_id'].replace('.MP4', '')
                    start_t = v_info['start_time']
                    unique_id = f"{video_id}_{start_t:.2f}".replace('.', '_')
                    
                    sig_path = os.path.join(output_dir, f"{unique_id}_sig.pt")
                    ego_path = os.path.join(output_dir, f"{unique_id}_ego.pt")
                    
                    # Skip if files already exist
                    # if os.path.exists(sig_path) and (not motion_model or os.path.exists(ego_path)):
                    #     continue
                    
                    # If the dataloader didn't provide tensors
                    if "siglip_tensor" not in batch:
                        continue

                    # --- Vision Extraction (Same as before) ---
                    sig_input = batch["siglip_tensor"][i:i+1].to(device, dtype=vision_model.dtype)
                    sig_feat = vision_model(sig_input).last_hidden_state.squeeze(0).cpu().to(torch.float16)

                    print(f"sig_feat {sig_feat.shape}")

                    torch.save(sig_feat, sig_path)
                    
                    # --- Motion Extraction (Modified to use Hook) ---
                    captured_features.clear() # Clear storage for this specific item
                    ego_input = batch["ego_tensor"][i:i+1].to(device, dtype=vision_model.dtype)
                    
                    # This triggers the hook_fn and fills captured_features
                    _ = motion_model.video_model.forward_features(ego_input)
                    
                    # Process the hooked 3D data: [1, 785, 768] -> [785, 768]
                    if captured_features:
                        ego_feat = captured_features[0].squeeze(0).to(torch.float16)
                        print(f"ego_feat {ego_feat.shape}")

                        torch.save(ego_feat, ego_path)
                    else:
                        raise RuntimeError(f"Hook failed to capture features for {unique_id}")
                        
                    # Explicitly clear GPU memory for this item
                    del sig_input
                    if motion_model: del ego_input
                        
                except Exception as e:
                    print(f"\n[Error] Skipping {unique_id}: {e}")
                    continue
            
            # Periodic garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # --- 2. Cleanup ---
    if hook_handle:
        hook_handle.remove()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

############################################################################################################################################

if __name__ == '__main__':

    current_file = os.path.basename(__file__)
    current_file = current_file.split(".")[0]

    EXTRACT_FEATURES = False
    LOAD_PRETRAINED_BRIDGE = False
    USE_PREEXTRACTED_FEATURES = True

    USE_MOTION = True

    TRAIN_BRIDGE = False
    VALIDATION_BRIDGE = False

    INFERENCE_BRIDGE = True
    VISUALIZE_FRAMES = True

    print(f"USE_MOTION: {USE_MOTION}")
    print(f"TRAIN_BRIDGE: {TRAIN_BRIDGE}")
    print(f"VALIDATION_BRIDGE: {VALIDATION_BRIDGE}")
    print(f"INFERENCE_BRIDGE: {INFERENCE_BRIDGE}")

    set_seed(42)
    os.makedirs("checkpoints", exist_ok=True)

    # --- 3. LOAD LLM ---
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                                quantization_config=bnb_config, device_map="auto",
                                                attn_implementation="sdpa") #  attn_implementation="flash_attention_2"
    llm.eval()

    for p in llm.parameters():
        p.requires_grad = False
    print("Loaded Qwen 2.5")
    llm.config.use_cache = False

    # potential LoRa adapters:
    # --- ADD LORA HERE ---
    print("Applying LoRA to the LLM...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Targeting more layers helps the LLM adapt to new modalities
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters() # This should show ~0.1% to 1% trainable params
    # We must explicitly set the LLM to train mode so the LoRA adapters track gradients
    llm.train() 
    llm.config.use_cache = False
    
    # --- 3. LOAD SIGLIP ---
    vision_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.float16).to("cuda")
    vision_model.eval()
    print("Loaded SigLIP")

    # --- 4. LOAD EGOVLP (FrozenInTime) ---
    print("Loading EgoVLP Motion Encoder...")
    EGO_REPO_PATH = os.path.abspath(os.path.join(BASE_DIR, "EgoVLPv2", "EgoVLPv2"))
    if EGO_REPO_PATH not in sys.path:
        sys.path.insert(0, EGO_REPO_PATH)

    original_cwd = os.getcwd()
    os.chdir(EGO_REPO_PATH)

    try:
        # PATCH: Import the custom RobertaModel and inject the missing attribute 
        # required by newer transformers versions.
        from model.roberta import RobertaModel
        if not hasattr(RobertaModel, 'all_tied_weights_keys'):
            RobertaModel.all_tied_weights_keys = property(lambda self: {})
        
        from model.model import FrozenInTime as EgoModel
    except ImportError as e:
        print(f"Import failed: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_cwd)

    # Config based on model.py requirements
    ego_config = {
        'video_params': {
            'model': 'SpaceTimeTransformer',
            'arch_config': 'base_patch16_224',
            'num_frames': 4,        # Changed from 8 to 4
            'pretrained': True
        },
        'text_params': {
            'model': 'roberta-base',
            'pretrained': True
        },
        'projection_dim': 4096,     # Changed from 256 to 4096
    }

    CHECKPOINT_PATH = r"C:\MultiModal\multiModel\pretrain_model\EgoVLPv2.pth"
    motion_model = load_egovlp(CHECKPOINT_PATH)
    motion_model.eval()


    # # Diagnostic to find the unpooled feature function
    # print(f"Model type: {type(motion_model.video_model)}")
    # # Test common EgoVLPv2 methods for unpooled output
    # with torch.no_grad():
    #     test_input = torch.randn(1, 3, 4, 224, 224).cuda().half() # Adjust for your setup
        
    #     # Try forward_features (common in timm-based models like EgoVLPv2)
    #     if hasattr(motion_model.video_model, 'forward_features'):
    #         out = motion_model.video_model.forward_features(test_input)
    #         print(f"forward_features shape: {out.shape}") # Looking for [1, 1569, 768] or similar

    #     # Check if the model has a patch_embed or blocks attribute
    #     print(f"Sub-modules: {list(motion_model.video_model._modules.keys())}")

    # sys.exit(0)

    print("Loaded EgoVLPv2")
    
    ####################

    # Get hidden dimensions
    llm_dim = llm.config.hidden_size             # 3584
    siglip_dim = vision_model.config.hidden_size # 1152
    egovlp_dim = 768                             # Unpooled token size

    print(f"{'LLM Hidden Size':<20} | {llm_dim:<10}")
    print(f"{'SigLIP (Vision)':<20} | {siglip_dim:<10}")
    print(f"{'EgoVLP (Video)':<20} | {egovlp_dim:<10}")

    bridge = MultimodalBridge(
        siglip_dim=siglip_dim,
        egovlp_dim=egovlp_dim,
        llm_dim=llm_dim,
        use_motion=USE_MOTION
    ).to("cuda").to(llm.dtype)

    resampler = PerceiverResampler(
        dim=llm_dim,
        num_latents=128,
        num_layers=2
    ).to("cuda").to(llm.dtype)
    
    temporal_resampler = TemporalPerceiverResampler(
        dim=llm_dim,
        num_latents=256,    # determines how many tokens it picks out, increase it if the model is consistently failing to identify small, crucial objects
        depth=2,
        num_heads=8
    ).to("cuda").to(llm.dtype)
    
    # Separate motion-specific bridge params from spatial params.
    # The motion pathway (egovlp_proj + motion_type_embed) starts from random init
    # against already-stable spatial features, so it gets a lower LR to avoid
    # dominating the gradient signal in early epochs.
    if USE_MOTION:
        motion_param_ids = set(
            id(p) for p in list(bridge.egovlp_proj.parameters()) + [bridge.motion_type_embed]
        )
        spatial_bridge_params = [p for p in bridge.parameters() if id(p) not in motion_param_ids]
        motion_bridge_params  = [p for p in bridge.parameters() if id(p) in motion_param_ids]
        bridge_param_groups = [
            {'params': spatial_bridge_params,  'lr': 1e-4},
            {'params': motion_bridge_params,   'lr': 2e-5},  # 5x lower: ramps up with more data seen
        ]
    else:
        bridge_param_groups = [{'params': bridge.parameters(), 'lr': 1e-4}]

    optimizer = torch.optim.AdamW(
        bridge_param_groups + [
            {'params': temporal_resampler.parameters(), 'lr': 1e-4},
            {'params': filter(lambda p: p.requires_grad, llm.parameters()), 'lr': 2e-5},
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    print("Preparing the training")
    
    siglip_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((384, 384)),
        T.ToTensor()
    ])

    egovlp_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    # --- 6. LOADING CSV ---
    # df = pd.read_csv("EPIC_KITCHENS_P01_captions.csv")

    # Initialize Dataset and Loader correctly
    CACHE_DIRECTORY = "extracted_features"
    dataset = EpicKitchensDataset(
        csv_path="EPIC_KITCHENS_P01_captions.csv", 
        siglip_transform=siglip_transform, 
        egovlp_transform=egovlp_transform, 
        history_len=4, 
        tokenizer=tokenizer,
        cache_dir=CACHE_DIRECTORY,
        dropout_prob=0.0
    )
    dataset_size = len(dataset)

    train_split = 0.1
    train_size = int(train_split * dataset_size)
    val_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    print("Training mode:")
    print("Vision:", True)
    print("Motion:", USE_MOTION)
    print("Text:", True)

    number_workers = 1

    train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=number_workers, # parallel video loading
    pin_memory=True, # faster GPU transfer
    persistent_workers=False,    # avoids worker restart every epoch
    prefetch_factor = 4, # workers preload batches, keeps GPU constantly fed
    collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=number_workers,
        pin_memory=True, # faster GPU transfer
        persistent_workers=False,    # avoids worker restart every epoch
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=number_workers,
        pin_memory=True, # faster GPU transfer
        persistent_workers=False,    # avoids worker restart every epoch
        collate_fn=collate_fn
    )

    if EXTRACT_FEATURES:
        dataset = EpicKitchensDataset(
        csv_path="EPIC_KITCHENS_P01_captions.csv", 
        siglip_transform=siglip_transform, 
        egovlp_transform=egovlp_transform, 
        history_len=4, 
        tokenizer=tokenizer,
        cache_dir=CACHE_DIRECTORY
        )
        # We run the extractor once using a fast dataloader with the raw dataset
        extract_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
        extract_and_save_features(
            dataloader=extract_loader, 
            output_dir=CACHE_DIRECTORY, 
            vision_model=vision_model, 
            motion_model=motion_model
        )
        print("Extraction complete! Set EXTRACT_FEATURES = False and run again to start training.")
        sys.exit(0)


    if USE_PREEXTRACTED_FEATURES:
        print("Pre-extracted features detected. Releasing Vision and Motion models from VRAM...")
        
        # 1. Move to CPU first (optional but safer)
        vision_model.cpu()
        motion_model.cpu()
        
        # 2. Delete the objects
        del vision_model
        del motion_model
        
        # 3. Force garbage collection and empty VRAM cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"VRAM Released. Current usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    if TRAIN_BRIDGE:
        
        print("-" * 30)
        print(f"Trainable Parameters in Bridge: {count_parameters(bridge):,}")
        print(f"Trainable Parameters in Resampler: {count_parameters(temporal_resampler):,}")
        print(f"Trainable Parameters in LLM (LoRA): {count_parameters(llm):,}")
        print(f"TOTAL Trainable Parameters: {count_parameters(bridge) + count_parameters(temporal_resampler) + count_parameters(llm):,}")
        print("-" * 30)

        if LOAD_PRETRAINED_BRIDGE:
            checkpoint_path = f"checkpoints/bridge_motion_True_main_20_1.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            bridge.load_state_dict(checkpoint["bridge_state_dict"])
            print(f"Successfully loaded pretrained bridge from {checkpoint_path}")

        best_loss = float('inf') # Track the best loss
        total_epochs = 8

        for epoch_full in range(total_epochs):
                        
            # --- FIX 2: Anti-shortcut dropout schedule ---
            # Previous schedule: 0% for epochs 0-1, then 30-50-30-0%.
            # Problem: the first 2 epochs of pure text training cemented the text shortcut
            # before the visual pathway had any chance to contribute. The LLM learned to
            # ignore visual tokens before they were ever useful.
            #
            # New schedule: start at 10% immediately (epoch 0), ramp to 70% peak at the
            # midpoint to maximally force visual reliance, then taper back to 10%.
            # This means the text shortcut is never fully available, so the LLM is forced
            # to extract some signal from visual tokens from the very first batch.
            #
            # Why NOT start at 1.0 as you asked: starting at 100% dropout means the model
            # has no text anchor at all in early epochs. The LLM has no way to learn what
            # good visual representations should predict — it just produces the most common
            # action as a prior. The visual pathway needs *some* text signal to calibrate
            # against, just not so much that the shortcut dominates.
            if epoch_full == 0:
                current_dropout = 0.10   # Epoch 0: small dropout — prevents text monopoly from day 1
            elif epoch_full == 1:
                current_dropout = 0.40   # Epochs 1-2: ramp up — visual pathway gets gradient
            elif epoch_full == 2:
                current_dropout = 0.40   # Epochs 3-4: peak — maximum visual forcing
            elif epoch_full == 3:
                current_dropout = 0.70
            elif epoch_full == 4:
                current_dropout = 0.70   # Epochs 5-6: taper — consolidation with mixed signal
            elif epoch_full == 5:
                current_dropout = 0.40
            elif epoch_full == 6:
                current_dropout = 0.40
            elif epoch_full == 7:
                current_dropout = 0.10  # Epoch 7: low dropout — final convergence with text
            else:
                current_dropout = 0.10   

            train_dataset.dataset.dropout_prob = current_dropout
            
            bridge.train()
            pbar = tqdm(train_loader, desc=f"Train Epoch {epoch_full+1}/{total_epochs}")

            # calculate average epoch loss
            epoch_loss_sum = 0.0
            num_batches = 0

            for batch in pbar:
                if batch is None: continue

                print(f"train_dataset.dataset.dropout_prob {train_dataset.dataset.dropout_prob}")

                # -------- Bridge Training (4-Step Temporal Window) --------
                optimizer.zero_grad()

                hist_sig = batch["history_sig"].to("cuda", dtype=llm.dtype) # [B, 4, 729, 1152]
                hist_ego = batch["history_ego"].to("cuda", dtype=llm.dtype) # [B, 4, 785, 768]

                all_visual_steps = []

                # Process each of the 4 time steps (t-4, t-3, t-2, t-1)
                for t_idx in range(4):
                    sig_t = hist_sig[:, t_idx, :, :] 
                    ego_t = hist_ego[:, t_idx, :, :]
                    
                    # Bridge processes this specific time step
                    # Result: [B, 1514, 3584]
                    step_embeds = bridge(sig_t, ego_t)

                    # # compress tokens # [B, 128, 3584]
                    # step_embeds = resampler(step_embeds)

                    all_visual_steps.append(step_embeds)

                # New shape: # Temporal compression
                # Temporarily concatenate the list to see the total token count
                # combined_tensor = torch.cat(all_visual_steps, dim=1)
                # print(f"before compression all_visual_steps_4step {combined_tensor.shape}")

                visual_embeds_4step = temporal_resampler(all_visual_steps)
                print(f"after compression visual_embeds_4step {visual_embeds_4step.shape}")

                print(batch["prompt_text"][0])

                # ... (standard tokenization for prompt/target remains the same) ...
                prompt_ids = tokenizer(batch["prompt_text"][0], return_tensors="pt").input_ids.to("cuda")
                target_ids = tokenizer(batch["target_text"][0], return_tensors="pt").input_ids.to("cuda")

                with torch.no_grad():
                    # p_embeds = llm.model.embed_tokens(prompt_ids)
                    # t_embeds = llm.model.embed_tokens(target_ids)

                    p_embeds = llm.base_model.model.model.embed_tokens(prompt_ids)
                    t_embeds = llm.base_model.model.model.embed_tokens(target_ids)

                # Combine 4 steps of video + prompt + target
                full_embeds = torch.cat([visual_embeds_4step, p_embeds, t_embeds], dim=1)

                # Update ignore_len to account for the massive 4-step sequence
                ignore_len = visual_embeds_4step.shape[1] + p_embeds.shape[1]
                labels = torch.full((1, full_embeds.shape[1]), -100, dtype=torch.long, device="cuda")
                labels[0, ignore_len:] = target_ids[0]

                outputs = llm(inputs_embeds=full_embeds, labels=labels)
                loss = outputs.loss
                loss.backward()
                # --- CHANGE 5: Gradient clipping ---
                # Prevents gradient spikes from corrupting LoRA adapters during early training
                # when bridge weights are random. Clips the global norm across ALL trainable params.
                all_params = (
                    list(bridge.parameters()) +
                    list(temporal_resampler.parameters()) +
                    [p for p in llm.parameters() if p.requires_grad]
                )
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

                pbar.set_postfix(loss=float(loss))

                # Accumulate loss for the epoch
                epoch_loss_sum += loss.item()
                num_batches += 1
                                 
            # ---------- Validation ----------
            if VALIDATION_BRIDGE:
                pass
            
            # Calculate average loss for this epoch
            avg_epoch_loss = epoch_loss_sum / max(num_batches, 1)
            print(f"\nEpoch {epoch_full+1} Average Loss: {avg_epoch_loss:.4f}")

            # ONLY SAVE IF THE LOSS IMPROVED
            print(f"Loss decreased from {best_loss:.4f} to {avg_epoch_loss:.4f}. Saving checkpoint...")
            best_loss = avg_epoch_loss # Update best loss

            checkpoint_name = f"checkpoints/bridge_motion_{USE_MOTION}_{str(train_split)}_{current_file}_{epoch_full}.pt"
            # torch.save({
            #     "bridge_state_dict": bridge.state_dict(),
            #     "temporal_resampler_state_dict": temporal_resampler.state_dict(), # Added this
            #     "optimizer_state_dict": optimizer.state_dict(),
            #     "epoch": epoch_full,
            # }, checkpoint_name)

            # Extract ONLY the trainable LoRA parameters
            from peft import get_peft_model_state_dict
            lora_state_dict = get_peft_model_state_dict(llm)

            torch.save({
                "bridge_state_dict": bridge.state_dict(),
                "temporal_resampler_state_dict": temporal_resampler.state_dict(), 
                "lora_state_dict": lora_state_dict, # <-- Added LoRA
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch_full,
            }, checkpoint_name)

            print("Final bridge saved.")
            
    # ---------- Inference ----------
    if INFERENCE_BRIDGE:
        print("\nRunning TEST inference...\n")
        if TRAIN_BRIDGE:
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = f"checkpoints/bridge_motion_True_0.1_main_t-4_lora_37_7.pt"

        checkpoint = torch.load(checkpoint_path, map_location="cuda")

        # Load Bridge and Resampler
        bridge.load_state_dict(checkpoint["bridge_state_dict"])
        temporal_resampler.load_state_dict(checkpoint["temporal_resampler_state_dict"])
        
        # Load LoRA Weights
        from peft import set_peft_model_state_dict
        if "lora_state_dict" in checkpoint:
            set_peft_model_state_dict(llm, checkpoint["lora_state_dict"])
            print("Successfully loaded LoRA adapters into LLM.")
        else:
            print("WARNING: No LoRA weights found in checkpoint!")

        bridge.eval()
        temporal_resampler.eval()
        llm.eval() # Switch LLM back to eval mode for generation

        print(f"Loaded full multimodal checkpoint: {checkpoint_path}")
        
        test_dropout = 0.0
        test_dataset.dataset.dropout_prob = test_dropout

        results = []
        for batch in tqdm(test_loader, desc="Testing"):

            if batch is None:
                continue

            prompt_text = batch["prompt_text"][0]
            target_text = batch["target_text"][0]

            video_info = batch["video_info"][0]
            video_path = video_info["video_path"]
            start_time = video_info["start_time"]
            end_time = video_info["end_time"]

            # -------- Check for History --------
            if "history_sig" not in batch:
                print(f"Skipping test batch: No cached features found.")
                continue
            
            # --- Pass the FULL 4-step history, not just index 3 ---
            hist_sig = batch["history_sig"].to("cuda", dtype=llm.dtype)
            
            if USE_MOTION:
                hist_ego = batch["history_ego"].to("cuda", dtype=llm.dtype)
            else:
                hist_ego = None

            with torch.no_grad():
                # Pass the full history sequences into run_inference
                predictions = run_inference(hist_sig, hist_ego, prompt_text, top_k=1)

            predictions = predictions[0]
            generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            results.append({
                "video_path": video_path,
                "start_time": start_time,
                "end_time": end_time,
                "prompt": prompt_text,
                "generated_captions": predictions,
                "target_caption": target_text,
                "generation_timestamp": generation_time,
                "model_checkpoint": checkpoint_path,
                "Motion": USE_MOTION
            })

            # Save CSV
            df_results = pd.DataFrame(results)

            output_file = f"test_inference_results_motion_{USE_MOTION}_{str(train_split)}_{current_file}.csv"
            df_results.to_csv(output_file, index=False)

            print(f"\nSaved inference results to: {output_file}")

    #### Analyze generated captions
    if INFERENCE_BRIDGE:
        df = pd.read_csv(output_file)
    else:
        output_file = f"test_inference_results_motion_True_main_23.csv"
        df = pd.read_csv(output_file)

    # Change column names if needed
    generated_col = "generated_captions"
    target_col = "target_caption"

    predictions = df[generated_col].astype(str).tolist()
    references = df[target_col].astype(str).tolist()

    # -------------------------------
    # Initialize scorers
    # -------------------------------
    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    meteor_scores = []
    rouge_scores = []

    # -------------------------------
    # Evaluation loop
    # -------------------------------
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):

        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())

        # BLEU scores
        bleu1 = sentence_bleu([ref_tokens], pred_tokens,
                            weights=(1,0,0,0),
                            smoothing_function=smooth)

        bleu2 = sentence_bleu([ref_tokens], pred_tokens,
                            weights=(0.5,0.5,0,0),
                            smoothing_function=smooth)

        bleu3 = sentence_bleu([ref_tokens], pred_tokens,
                            weights=(0.33,0.33,0.33,0),
                            smoothing_function=smooth)

        bleu4 = sentence_bleu([ref_tokens], pred_tokens,
                            weights=(0.25,0.25,0.25,0.25),
                            smoothing_function=smooth)

        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        bleu4_scores.append(bleu4)

        # METEOR (requires tokenized input)
        meteor = meteor_score([ref_tokens], pred_tokens)
        meteor_scores.append(meteor)

        # ROUGE-L
        rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure
        rouge_scores.append(rouge_l)

    # -------------------------------
    # Print average results
    # -------------------------------
    print("\nEvaluation Results:")
    print(f"BLEU-1 : {sum(bleu1_scores)/len(bleu1_scores):.4f}")
    print(f"BLEU-2 : {sum(bleu2_scores)/len(bleu2_scores):.4f}")
    print(f"BLEU-3 : {sum(bleu3_scores)/len(bleu3_scores):.4f}")
    print(f"BLEU-4 : {sum(bleu4_scores)/len(bleu4_scores):.4f}")
    print(f"METEOR : {sum(meteor_scores)/len(meteor_scores):.4f}")
    print(f"ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.4f}")

    df["BLEU-1"] = bleu1_scores
    df["BLEU-2"] = bleu2_scores
    df["BLEU-3"] = bleu3_scores
    df["BLEU-4"] = bleu4_scores
    df["METEOR"] = meteor_scores
    df["ROUGE-L"] = rouge_scores

    df.to_csv(f"analysis_{output_file}", index=False)

    # -------------------------------
    # Heatmap Visualizations
    # -------------------------------
    if VISUALIZE_FRAMES:
        if TRAIN_BRIDGE:
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = f"checkpoints/bridge_motion_True_0.1_main_t-4_lora_37_7.pt"

        checkpoint = torch.load(checkpoint_path, map_location="cuda")

        # Load Bridge and Resampler
        bridge.load_state_dict(checkpoint["bridge_state_dict"])
        temporal_resampler.load_state_dict(checkpoint["temporal_resampler_state_dict"])
        
        # Load LoRA Weights
        from peft import set_peft_model_state_dict
        if "lora_state_dict" in checkpoint:
            set_peft_model_state_dict(llm, checkpoint["lora_state_dict"])
            print("Successfully loaded LoRA adapters into LLM.")
        else:
            print("WARNING: No LoRA weights found in checkpoint!")

        bridge.eval()
        temporal_resampler.eval()
        llm.eval() # Switch LLM back to eval mode for generation

        from visualize_modalities import run_modality_visualizations
        run_modality_visualizations(
            dataloader         = test_loader,
            bridge             = bridge,
            temporal_resampler = temporal_resampler,
            llm                = llm,
            tokenizer          = tokenizer,
            use_motion         = USE_MOTION,
            n_samples          = 5,
            save_dir           = "modality_visualizations"
        )