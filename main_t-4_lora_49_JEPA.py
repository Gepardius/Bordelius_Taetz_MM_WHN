"""
JEPA-Aligned Multimodal Architecture for Kitchen Action Prediction
===================================================================
Goal  : predict the t-0 caption (text) from t-4 … t-1 context
         (vision + motion + text per timestep).

JEPA changes vs lora_47:
  - Modality collapse fix  : text embeddings are GATED (trained gate scalar per
                             modality), so the model cannot trivially short-circuit
                             on text alone. The gate starts closed for text and
                             open for vision/motion, forcing the visual pathway to
                             contribute gradient before text is reintroduced.
  - Context encoder        : TemporalPerceiverResampler (unchanged structure) now
                             acts as the JEPA *context encoder* s_x = f_θ(observed).
  - Target encoder         : EMA copy of the context encoder operating on the t-0
                             features. Gradients do NOT flow through the target.
                             Updated via exponential moving average each step.
  - Predictor              : Lightweight Transformer that maps s_x + a mask token
                             to a predicted representation s_ŷ in latent space.
  - JEPA loss              : L2 between s_ŷ and stop_gradient(s_y) in latent space.
                             No pixel/token reconstruction — purely representational.
  - Inference              : predictor output s_ŷ replaces the resampler latents
                             that are fed into Qwen. The LLM decodes from the
                             predicted latent, not from the context encoder directly.

Modality gating (anti-collapse):
  - ModalityGate learns a scalar α ∈ [0,1] per modality.
  - α_text starts near 0; α_vision/motion start near 1.
  - Gate value is sigmoid(learnable_logit).
  - LLM text embeddings of history actions are multiplied by α_text before
    concatenation, so the model cannot rely on them when α_text ≈ 0.
  - Dropout schedule is REMOVED — gating replaces it structurally.
"""

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
import copy

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import torch._dynamo
from peft import LoraConfig, get_peft_model

import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm

import matplotlib.pyplot as plt
from torchvision import transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA


torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad")


# ---------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# ---------------------------------------------------------------------------
os.environ["HF_HOME"] = "C:/MultiModal/hf_cache"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


# ===========================================================================
# 2. MODALITY GATE  (anti-collapse)
# ===========================================================================
class ModalityGate(nn.Module):
    """
    Learnable scalar gate per modality: α = sigmoid(logit).

    init_logit controls the starting gate value:
      +5  → α ≈ 0.993  (nearly open)
      -5  → α ≈ 0.007  (nearly closed)

    Usage: gate the text embedding before concatenation so the model must
    learn from vision/motion first.  The gate is differentiable — the model
    can open the text gate once it no longer needs it as a shortcut.
    """
    def __init__(self, init_logit_vision: float = 5.0,
                       init_logit_motion: float = 5.0,
                       init_logit_text:   float = -5.0):
        super().__init__()
        self.logit_vision = nn.Parameter(torch.tensor(init_logit_vision))
        self.logit_motion = nn.Parameter(torch.tensor(init_logit_motion))
        self.logit_text   = nn.Parameter(torch.tensor(init_logit_text))

    @property
    def alpha_vision(self): return torch.sigmoid(self.logit_vision)
    @property
    def alpha_motion(self): return torch.sigmoid(self.logit_motion)
    @property
    def alpha_text(self):   return torch.sigmoid(self.logit_text)

    def gate_vision(self, x): return x * self.alpha_vision
    def gate_motion(self, x): return x * self.alpha_motion
    def gate_text(self,   x): return x * self.alpha_text

    def extra_repr(self):
        return (f"α_vision={self.alpha_vision.item():.3f}  "
                f"α_motion={self.alpha_motion.item():.3f}  "
                f"α_text={self.alpha_text.item():.3f}")


# ===========================================================================
# 2b. MODALITY ALIGNMENT PROJECTOR
# ===========================================================================
class ModalityAlignmentProjector(nn.Module):
    """
    Lightweight 2-layer MLP that maps EgoVLP (motion) and text tokens into
    the same subspace as SigLIP (vision) tokens BEFORE they enter the context
    encoder.

    Problem it solves (observed in lora_47/48):
      PCA of context-encoder input tokens showed EGO and TXT tokens landing
      at PC1≈40 — far from the dense SigLIP cloud centred near PC1≈0.
      The Perceiver latents therefore attend almost exclusively to SigLIP
      (99.3% of attention, Fig 1), because the motion and text tokens are
      spatially isolated in the shared embedding space.

    Fix:
      Project EGO and TXT tokens through a learned affine + LayerNorm that
      pulls them into the same norm/scale regime as the projected SigLIP tokens
      (which come out of bridge.siglip_proj with mean≈0, std≈0.1).
      Vision tokens are passed through unchanged — they are already in the
      correct subspace.

    Usage (in training loop, before context_encoder call):
        ego_aligned = modality_align.align_motion(mot_tok)  # [B, N_mot, D]
        txt_aligned = modality_align.align_text(txt_tok)    # [B, T_txt, D]
      Then concatenate as normal: fused = cat([vis_tok, ego_aligned], dim=1)
      and pass txt_aligned to text_embeds_list in the encoder.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Shared hidden size — small enough to stay fast, large enough to rotate
        hidden = dim // 4  # 896 for llm_dim=3584

        self.motion_proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )

    def align_motion(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N_mot, D] → [B, N_mot, D]  (aligned to SigLIP subspace)"""
        return self.motion_proj.float()(x.float()).to(x.dtype)

    def align_text(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T_txt, D] → [B, T_txt, D]  (aligned to SigLIP subspace)"""
        return self.text_proj.float()(x.float()).to(x.dtype)




# ===========================================================================
# 3. MULTIMODAL BRIDGE  (unchanged from lora_47 except text gating injection)
# ===========================================================================
class MultimodalBridge(nn.Module):
    """
    Projects SigLIP (vision) + EgoVLP (motion) tokens into LLM space.
    Text is NOT processed here — it is handled separately in the encoder and
    gated by ModalityGate before being concatenated with vision/motion tokens.

    Token count per timestep (USE_MOTION=True, top_k=8):
        Vision first frame : 729
        Vision last frame  : 729
        Motion CLS         :   1
        Motion intra delta :   1
        Motion top-k patch :   8
        Total              : 1468
    """
    def __init__(self, siglip_dim=1152, egovlp_dim=768, llm_dim=3584,
                 use_motion=True, top_k_patches=8):
        super().__init__()
        self.use_motion    = use_motion
        self.top_k_patches = top_k_patches

        self.siglip_proj = nn.Sequential(
            nn.Linear(siglip_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        self.vision_change_embed = nn.Parameter(torch.zeros(1, 1, llm_dim))

        if self.use_motion:
            self.egovlp_proj = nn.Sequential(
                nn.Linear(egovlp_dim, llm_dim),
                nn.LayerNorm(llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim)
            )
            self.motion_cls_embed         = nn.Parameter(torch.zeros(1, 1, llm_dim))
            self.motion_intra_delta_embed = nn.Parameter(torch.zeros(1, 1, llm_dim))
            self.motion_topk_rank_embeds  = nn.Parameter(
                torch.zeros(top_k_patches, 1, llm_dim))
            self._ego_last_patch_start = 1 + 3 * 196   # 589
            self._ego_last_patch_end   = 1 + 4 * 196   # 785
            self.cross_modal_gate = nn.MultiheadAttention(
                embed_dim=llm_dim, num_heads=4, batch_first=True)
            self.cross_modal_norm = nn.LayerNorm(llm_dim)

    def forward(self, sig_first, sig_last, ego_embeds=None):
        orig_dtype = sig_first.dtype
        vis_first = self.siglip_proj.float()(sig_first.float()).to(orig_dtype)
        vis_last  = self.siglip_proj.float()(sig_last.float()).to(orig_dtype)
        vis_last  = vis_last + self.vision_change_embed
        vis_tokens = torch.cat([vis_first, vis_last], dim=1)

        if self.use_motion and ego_embeds is not None:
            s, e = self._ego_last_patch_start, self._ego_last_patch_end
            cls_raw  = ego_embeds[:, 0:1, :]
            mot_cls  = self.egovlp_proj.float()(cls_raw.float()).to(orig_dtype)
            mot_cls  = mot_cls + self.motion_cls_embed

            last_patches_raw  = ego_embeds[:, s:e, :]
            first_patches_raw = ego_embeds[:, 1:197, :]
            patch_norms = last_patches_raw.float().norm(dim=-1)
            k = min(self.top_k_patches, last_patches_raw.shape[1])
            topk_idx = patch_norms.topk(k, dim=-1).indices
            idx_exp  = topk_idx.unsqueeze(-1).expand(-1, -1, ego_embeds.shape[-1])
            top_last  = torch.gather(last_patches_raw,  dim=1, index=idx_exp)
            top_first = torch.gather(first_patches_raw, dim=1, index=idx_exp)

            intra_delta_raw = (top_last - top_first).mean(dim=1, keepdim=True)
            mot_intra = self.egovlp_proj.float()(intra_delta_raw.float()).to(orig_dtype)
            mot_intra = mot_intra + self.motion_intra_delta_embed

            topk_proj   = self.egovlp_proj.float()(top_last.float()).to(orig_dtype)
            rank_embeds = self.motion_topk_rank_embeds.permute(1, 0, 2)
            topk_proj   = topk_proj + rank_embeds

            mot_tokens = torch.cat([mot_cls, mot_intra, topk_proj], dim=1)
            gate_in    = self.cross_modal_norm(mot_tokens)
            gated, _   = self.cross_modal_gate(
                query=gate_in, key=vis_tokens, value=vis_tokens)
            mot_tokens = mot_tokens + gated
            return torch.cat([vis_tokens, mot_tokens], dim=1)

        return vis_tokens


# ===========================================================================
# 4. JEPA CONTEXT ENCODER  (TemporalPerceiverResampler, enriched with text)
# ===========================================================================
class JEPAContextEncoder(nn.Module):
    """
    Encodes the 4-step observation context into a fixed-size latent s_x.

    Changes vs TemporalPerceiverResampler in lora_47:
      - Accepts an optional text_tokens argument [B, num_steps, T_text, D].
        Each timestep's text embedding is gated by ModalityGate.alpha_text and
        concatenated with the corresponding vision/motion tokens BEFORE the
        change-attention step.  This preserves the temporal change-attention
        benefit while giving text a chance to contribute once the gate opens.
      - The modality_gate is passed in (shared with predictor and LLM bridge)
        so all three modules see the same gate values.
    """
    def __init__(self, dim, num_latents=128, depth=2, num_heads=8, num_steps=4):
        super().__init__()
        self.num_steps = num_steps
        self.latents   = nn.Parameter(torch.randn(num_latents, dim))
        self.time_embed = nn.Parameter(torch.randn(num_steps, 1, dim))

        self.change_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.change_norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=dim, num_heads=num_heads, batch_first=True),
                "norm1": nn.LayerNorm(dim),
                "ff": nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            }) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, step_embeds_list, text_embeds_list=None, modality_gate=None):
        """
        step_embeds_list  : list of num_steps tensors [B, N_vis, D]
                            ordered t-4, t-3, t-2, t-1
        text_embeds_list  : list of num_steps tensors [B, T_text, D] or None
        modality_gate     : ModalityGate instance or None

        Returns s_x : [B, num_latents, D]
        """
        B = step_embeds_list[0].shape[0]

        stamped = []
        for t_idx, step in enumerate(step_embeds_list):
            tokens = step + self.time_embed[t_idx]

            # Fuse gated text tokens for this timestep
            if text_embeds_list is not None and modality_gate is not None:
                txt = text_embeds_list[t_idx]          # [B, T_text, D]
                txt = modality_gate.gate_text(txt)      # multiply by α_text
                tokens = torch.cat([tokens, txt], dim=1)

            stamped.append(tokens)

        # Change attention: each step attends to t-4 as reference.
        # Use len(stamped) — not self.num_steps — so this works both when
        # the context encoder receives 4 steps AND when the target encoder
        # receives a repeated single-step list of any length.
        reference = stamped[0]
        changed   = [reference]
        for t_idx in range(1, len(stamped)):
            query_in = self.change_norm(stamped[t_idx])
            delta, _ = self.change_attn(
                query=query_in, key=reference, value=reference)
            changed.append(stamped[t_idx] + delta)

        x = torch.cat(changed, dim=1)  # [B, num_steps * N_tokens, D]

        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            cross_in   = layer["norm1"](latents)
            attn_out, _ = layer["cross_attn"](
                query=cross_in, key=x, value=x)
            latents = latents + attn_out
            latents = latents + layer["ff"](latents)

        return self.norm(latents)  # s_x : [B, num_latents, D]


# ===========================================================================
# 5. JEPA PREDICTOR
# ===========================================================================
class JEPAPredictor(nn.Module):
    """
    Predicts the target representation s_ŷ from the context s_x.

    Architecture: narrow Transformer (dim → predictor_dim → dim) that takes
    s_x as context tokens and a single learnable mask token as the query for
    the position to predict (t-0).

    Intentionally smaller than the context encoder — the predictor must
    compress, not memorise. Using the same capacity would let the predictor
    ignore s_x and reconstruct s_y from texture statistics.

    Output s_ŷ has the same shape as s_y: [B, num_latents, D].
    """
    def __init__(self, dim, predictor_dim=512, num_latents=128,
                 depth=4, num_heads=8):
        super().__init__()
        self.num_latents = num_latents

        # Projection into predictor space (narrower bottleneck)
        self.input_proj  = nn.Linear(dim, predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, dim)

        # Learnable mask token representing the t-0 query
        self.mask_token = nn.Parameter(torch.randn(1, num_latents, predictor_dim))

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(
                    embed_dim=predictor_dim, num_heads=num_heads, batch_first=True),
                "norm0": nn.LayerNorm(predictor_dim),
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=predictor_dim, num_heads=num_heads, batch_first=True),
                "norm1": nn.LayerNorm(predictor_dim),
                "ff": nn.Sequential(
                    nn.LayerNorm(predictor_dim),
                    nn.Linear(predictor_dim, predictor_dim * 4),
                    nn.GELU(),
                    nn.Linear(predictor_dim * 4, predictor_dim)
                )
            }) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)

    def forward(self, s_x):
        """
        s_x : [B, num_latents, D]   — context encoder output
        Returns s_ŷ : [B, num_latents, D]  — predicted target representation
        """
        B = s_x.shape[0]

        # Project context into predictor space
        ctx = self.input_proj(s_x)              # [B, num_latents, P]

        # Mask tokens are the "questions" the predictor must answer
        queries = self.mask_token.expand(B, -1, -1)   # [B, num_latents, P]

        for layer in self.layers:
            # Self-attention among the mask queries
            q_norm  = layer["norm0"](queries)
            sa_out, _ = layer["self_attn"](q_norm, q_norm, q_norm)
            queries = queries + sa_out

            # Cross-attention: queries read from the context
            q_norm  = layer["norm1"](queries)
            ca_out, _ = layer["cross_attn"](
                query=q_norm, key=ctx, value=ctx)
            queries = queries + ca_out
            queries = queries + layer["ff"](queries)

        queries = self.norm(queries)
        return self.output_proj(queries)         # s_ŷ : [B, num_latents, D]


# ===========================================================================
# 6. EMA TARGET ENCODER WRAPPER
# ===========================================================================
class EMATargetEncoder(nn.Module):
    """
    Exponential-moving-average copy of JEPAContextEncoder.

    The target encoder produces s_y = f_ξ(t-0 features) where ξ are updated
    via: ξ ← τ·ξ + (1−τ)·θ  (θ = online context encoder params).

    Gradients never flow through this module.  Call .update(online_encoder)
    once per training step after the main backward.
    """
    def __init__(self, online_encoder: JEPAContextEncoder, tau: float = 0.996):
        super().__init__()
        self.tau     = tau
        self.encoder = copy.deepcopy(online_encoder)
        # Freeze all params — EMA updates are done manually
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, online_encoder: JEPAContextEncoder):
        """Call once per step: ξ ← τ·ξ + (1−τ)·θ"""
        for ξ, θ in zip(self.encoder.parameters(),
                        online_encoder.parameters()):
            ξ.data.mul_(self.tau).add_(θ.data, alpha=1.0 - self.tau)

    @torch.no_grad()
    def encode_target(self, step_embeds_list, text_embeds_list=None,
                      modality_gate=None):
        """Encode t-0 features; no gradient."""
        return self.encoder(step_embeds_list, text_embeds_list, modality_gate)


# ===========================================================================
# 7. JEPA LOSS
# ===========================================================================
def jepa_loss(s_hat: torch.Tensor, s_y: torch.Tensor,
              normalize: bool = True) -> torch.Tensor:
    """
    L2 loss in latent space between predicted s_ŷ and stop-gradient target s_y.

    normalize=True: normalise both to unit sphere before computing MSE.
    This prevents the trivial solution where both collapse to zero.
    """
    if normalize:
        s_hat = nn.functional.normalize(s_hat.float(), dim=-1)
        s_y   = nn.functional.normalize(s_y.float(),   dim=-1)
    loss = (s_hat.float() - s_y.float()).pow(2).mean()
    return loss


# ===========================================================================
# 8. DATASET  (unchanged structure — adds text embedding extraction)
# ===========================================================================
class EpicKitchensDataset(Dataset):
    def __init__(self, csv_path, siglip_transform, egovlp_transform,
                 history_len=4, tokenizer=None, cache_dir=None,
                 extraction_mode=False):
        self.cache_dir        = cache_dir
        df                    = pd.read_csv(csv_path)
        self.data             = df.to_dict('records')
        self.siglip_transform = siglip_transform
        self.egovlp_transform = egovlp_transform
        self.history_len      = history_len
        self.tokenizer        = tokenizer
        self.extraction_mode  = extraction_mode

    def __len__(self):
        return len(self.data) - self.history_len

    def __getitem__(self, idx):
        actual_idx    = idx + self.history_len
        target_sample = self.data[actual_idx]
        target_text   = target_sample['narration']

        if self.extraction_mode:
            vid_path  = target_sample.get("location", "")
            start_t   = target_sample.get("start_seconds", 0)
            end_t     = target_sample.get("stop_seconds", start_t + 1)
            frames_first, frames_last, frames_ego = load_video_segment(
                vid_path, start_t, end_t)
            if frames_first is None:
                return None
            sig_first_tensor = self.siglip_transform(frames_first)
            sig_last_tensor  = self.siglip_transform(frames_last)
            if frames_ego and len(frames_ego) == 4:
                ego_tensor = torch.stack(
                    [self.egovlp_transform(f) for f in frames_ego], dim=0)
            else:
                ego_tensor = torch.zeros(4, 3, 224, 224)
            return {
                "siglip_first_tensor": sig_first_tensor,
                "siglip_last_tensor":  sig_last_tensor,
                "ego_tensor":          ego_tensor,
                "prompt_text":         "",
                "target_text":         target_text,
                "video_info": {
                    "video_path": vid_path,
                    "start_time": start_t,
                    "end_time":   end_t,
                },
            }

        # Gather history
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
        prev_actions.reverse()

        # Prompt (no dropout — replaced by ModalityGate)
        system_msg = (
            "You are an AI assistant that predicts the next action in "
            "egocentric cooking videos. "
            "Output ONLY a 2-3 word action phrase in the format '[verb] [noun]'. "
            "Examples: 'take knife', 'pour oil', 'close fridge'. "
            "Never chain multiple actions. Never explain."
        )
        user_msg = (
            f"Previous actions:\n"
            f"t-4: {prev_actions[0]}\n"
            f"t-3: {prev_actions[1]}\n"
            f"t-2: {prev_actions[2]}\n"
            f"t-1: {prev_actions[3]}\n\n"
            f"Predict the next action:"
        )
        prompt_text = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Load cached features for context steps t-4 … t-1
        history_sig_first = []
        history_sig_last  = []
        history_ego       = []
        history_texts     = []

        for i in range(self.history_len, 0, -1):
            prev_idx    = actual_idx - i
            prev_sample = self.data[prev_idx]
            vid_id      = os.path.basename(prev_sample["location"]).replace('.MP4', '')
            start_t     = prev_sample["start_seconds"]
            cache_file  = os.path.join(self.cache_dir, f"{vid_id}_{start_t:.2f}.pt")

            if os.path.exists(cache_file):
                data = torch.load(cache_file)
                if "sig_first_features" in data and "sig_last_features" in data:
                    sig_first = data["sig_first_features"].squeeze(0)
                    sig_last  = data["sig_last_features"].squeeze(0)
                else:
                    sig_first = data["sig_features"].squeeze(0)
                    sig_last  = sig_first
                ego = data["ego_features"].squeeze(0) if data["ego_features"] is not None \
                      else torch.zeros((785, 768))
            else:
                sig_first = torch.zeros((729, 1152))
                sig_last  = torch.zeros((729, 1152))
                ego       = torch.zeros((785, 768))

            history_sig_first.append(sig_first)
            history_sig_last.append(sig_last)
            history_ego.append(ego)
            history_texts.append(prev_sample["narration"])

        # Load t-0 cached features for JEPA target encoder
        vid_id_t0    = os.path.basename(target_sample["location"]).replace('.MP4', '')
        start_t0     = target_sample["start_seconds"]
        cache_t0     = os.path.join(self.cache_dir, f"{vid_id_t0}_{start_t0:.2f}.pt")

        if os.path.exists(cache_t0):
            data_t0 = torch.load(cache_t0)
            if "sig_first_features" in data_t0 and "sig_last_features" in data_t0:
                t0_sig_first = data_t0["sig_first_features"].squeeze(0)
                t0_sig_last  = data_t0["sig_last_features"].squeeze(0)
            else:
                t0_sig_first = data_t0["sig_features"].squeeze(0)
                t0_sig_last  = t0_sig_first
            t0_ego = data_t0["ego_features"].squeeze(0) \
                     if data_t0["ego_features"] is not None \
                     else torch.zeros((785, 768))
        else:
            t0_sig_first = torch.zeros((729, 1152))
            t0_sig_last  = torch.zeros((729, 1152))
            t0_ego       = torch.zeros((785, 768))

        return {
            # Context (4 steps)
            "history_sig_first": torch.stack(history_sig_first),   # [4, 729, 1152]
            "history_sig_last":  torch.stack(history_sig_last),    # [4, 729, 1152]
            "history_ego":       torch.stack(history_ego),         # [4, 785, 768]
            "history_texts":     history_texts,
            # Target (t-0) — used by target encoder during training
            "t0_sig_first":      t0_sig_first,                     # [729, 1152]
            "t0_sig_last":       t0_sig_last,                      # [729, 1152]
            "t0_ego":            t0_ego,                           # [785, 768]
            # Text supervision
            "prompt_text":       prompt_text,
            "target_text":       target_text,
            "video_info": {
                "video_path": target_sample.get("location", ""),
                "start_time": target_sample.get("start_seconds", 0),
                "end_time":   target_sample.get("stop_seconds", 0),
            },
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    if "siglip_first_tensor" in batch[0]:
        return {
            "siglip_first_tensor": torch.stack([b["siglip_first_tensor"] for b in batch]),
            "siglip_last_tensor":  torch.stack([b["siglip_last_tensor"]  for b in batch]),
            "ego_tensor":          torch.stack([b["ego_tensor"]          for b in batch]),
            "prompt_text":  [b["prompt_text"]  for b in batch],
            "target_text":  [b["target_text"]  for b in batch],
            "video_info":   [b["video_info"]   for b in batch],
        }
    return {
        "history_sig_first": torch.stack([b["history_sig_first"] for b in batch]),
        "history_sig_last":  torch.stack([b["history_sig_last"]  for b in batch]),
        "history_ego":       torch.stack([b["history_ego"]       for b in batch]),
        "t0_sig_first":      torch.stack([b["t0_sig_first"]      for b in batch]),
        "t0_sig_last":       torch.stack([b["t0_sig_last"]       for b in batch]),
        "t0_ego":            torch.stack([b["t0_ego"]            for b in batch]),
        "history_texts":  [b["history_texts"]  for b in batch],
        "prompt_text":    [b["prompt_text"]    for b in batch],
        "target_text":    [b["target_text"]    for b in batch],
        "video_info":     [b["video_info"]     for b in batch],
    }


# ===========================================================================
# 9. VIDEO UTILITIES  (unchanged)
# ===========================================================================
def load_video_segment(video_path, start_sec, end_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, []
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, min(int(start_sec * fps), total_frames - 1))
    end_frame   = max(start_frame + 1, min(int(end_sec * fps), total_frames))
    siglip_first_index = start_frame
    siglip_last_index  = max(start_frame, end_frame - 1)
    ego_indices        = np.linspace(start_frame, end_frame - 1, 4).astype(int)
    all_indices        = sorted(set(ego_indices.tolist() + [siglip_first_index, siglip_last_index]))
    frames_siglip_first = None
    frames_siglip_last  = None
    frames_ego_dict     = {}
    for frame_idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_idx == siglip_first_index:
            frames_siglip_first = frame
        if frame_idx == siglip_last_index:
            frames_siglip_last = frame
        if frame_idx in set(ego_indices.tolist()):
            frames_ego_dict[frame_idx] = frame
    cap.release()
    if frames_siglip_first is None:
        return None, None, []
    if frames_siglip_last is None:
        frames_siglip_last = frames_siglip_first
    frames_ego = []
    for idx in ego_indices:
        if idx in frames_ego_dict:
            frames_ego.append(frames_ego_dict[idx])
        elif frames_ego_dict:
            closest = min(frames_ego_dict.keys(), key=lambda k: abs(k - idx))
            frames_ego.append(frames_ego_dict[closest])
    return frames_siglip_first, frames_siglip_last, frames_ego


def load_egovlp(checkpoint_path):
    model = EgoModel(
        video_params=ego_config['video_params'],
        text_params=ego_config['text_params'],
        projection_dim=ego_config['projection_dim']
    )
    checkpoint  = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict  = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    new_sd      = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=False)
    return model.cuda().half().eval()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global seed set to: {seed}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# 10. INFERENCE
# ===========================================================================
def run_inference(hist_sig_first, hist_sig_last, hist_ego, prompt_text, top_k=3):
    """
    During inference the target encoder is not available.
    The predictor output s_ŷ stands in for s_y — the LLM decodes from the
    predicted latent representation.
    """
    bridge.eval()
    context_encoder.eval()
    predictor.eval()
    llm.eval()

    tokens       = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    prompt_ids   = tokens.input_ids
    prompt_mask  = tokens.attention_mask

    with torch.no_grad():
        prompt_embeds = llm.base_model.model.model.embed_tokens(prompt_ids)

        # Build text embeds for each context step
        text_embeds_list = []
        for t_idx in range(4):
            # Use a simple zero embedding at inference (history text already in prompt)
            text_embeds_list.append(
                torch.zeros(1, 1, llm.config.hidden_size,
                            device="cuda", dtype=llm.dtype))

        # Context encoder: encode 4-step history with modality alignment
        all_visual_steps = []
        for t_idx in range(4):
            sf  = hist_sig_first[:, t_idx, :, :].to(llm.dtype)
            sl  = hist_sig_last[:, t_idx, :, :].to(llm.dtype)
            ego = hist_ego[:, t_idx, :, :].to(llm.dtype) if USE_MOTION else None
            step_vis = bridge(sf, sl, ego)
            # At inference the bridge returns vis+mot concatenated; we apply
            # alignment to the motion portion. Since we cannot easily split here,
            # we pass the full fused tensor through the full bridge and gate, then
            # align only the motion tokens inline.
            if USE_MOTION and ego is not None:
                n_vis = 1458  # 729 first + 729 last frame tokens
                vis_part = step_vis[:, :n_vis, :]
                mot_part = step_vis[:, n_vis:, :]
                vis_part = modality_gate.gate_vision(vis_part)
                mot_part = modality_gate.gate_motion(mot_part)
                mot_part = modality_align.align_motion(mot_part)
                step_vis = torch.cat([vis_part, mot_part], dim=1)
            else:
                step_vis = modality_gate.gate_vision(step_vis)
            all_visual_steps.append(step_vis)
        # Align text zero-embeddings (keeps encoder consistent with training)
        text_embeds_list = [modality_align.align_text(t) for t in text_embeds_list]

        s_x   = context_encoder(all_visual_steps, text_embeds_list, modality_gate)
        # Predictor: predict t-0 representation
        s_hat = predictor(s_x)                  # [B, num_latents, D]

        # Decode from predicted latent
        inputs_embeds = torch.cat([s_hat, prompt_embeds], dim=1)
        visual_mask   = torch.ones(s_hat.shape[:2],
                                   dtype=prompt_mask.dtype, device="cuda")
        attention_mask = torch.cat([visual_mask, prompt_mask], dim=1)

        output_ids = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=4,   # targets avg 2.7 words; 8 caused over-generation
            do_sample=False,
            num_beams=top_k,
            num_return_sequences=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=-2.0,
            no_repeat_ngram_size=3,
        )

        generated_texts = []
        for i in range(top_k):
            text = tokenizer.decode(output_ids[i], skip_special_tokens=True).strip()
            text = text.split('\n')[0]
            text = text.split('Human:')[0].split('Assistant:')[0]
            for connector in [' and ', ', and ', '. ', ',']:
                if connector in text:
                    text = text.split(connector)[0].strip()
                    break
            generated_texts.append(text.strip().rstrip('.,'))

    return generated_texts


# ===========================================================================
# 11. FEATURE EXTRACTION  (unchanged from lora_47)
# ===========================================================================
def extract_and_save_features(dataloader, output_dir, vision_model,
                               motion_model, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    vision_model.eval()
    hook_handle       = None
    captured_features = []

    if motion_model:
        motion_model.eval()
        def hook_fn(module, input, output):
            captured_features.append(output.detach().cpu())
        hook_handle = motion_model.video_model.norm.register_forward_hook(hook_fn)

    print(f"\nStarting Feature Extraction to '{output_dir}'...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            if batch is None:
                continue
            for i in range(len(batch["prompt_text"])):
                v_info     = batch["video_info"][i]
                video_path = v_info["video_path"]
                video_id   = os.path.basename(video_path).replace('.MP4', '').replace('.mp4', '')
                start_t    = float(v_info["start_time"])
                unique_id  = f"{video_id}_{start_t:.2f}"
                cache_path = os.path.join(output_dir, f"{unique_id}.pt")
                if os.path.exists(cache_path):
                    continue
                if "siglip_first_tensor" not in batch:
                    continue
                sig_first_input = batch["siglip_first_tensor"][i:i+1].to(device, dtype=vision_model.dtype)
                sig_last_input  = batch["siglip_last_tensor"][i:i+1].to(device, dtype=vision_model.dtype)
                sig_first_feat  = vision_model(sig_first_input).last_hidden_state.squeeze(0).cpu().to(torch.float16)
                sig_last_feat   = vision_model(sig_last_input).last_hidden_state.squeeze(0).cpu().to(torch.float16)
                del sig_first_input, sig_last_input
                ego_feat = None
                if motion_model:
                    captured_features.clear()
                    ego_input = batch["ego_tensor"][i:i+1].to(device, dtype=vision_model.dtype)
                    _ = motion_model.video_model.forward_features(ego_input)
                    del ego_input
                    if captured_features:
                        ego_feat = captured_features[0].squeeze(0).to(torch.float16)
                    else:
                        raise RuntimeError("Hook failed to capture EgoVLP features")
                torch.save({
                    "sig_first_features": sig_first_feat,
                    "sig_last_features":  sig_last_feat,
                    "sig_features":       sig_last_feat,
                    "ego_features":       ego_feat,
                }, cache_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if hook_handle:
        hook_handle.remove()

# ------------------------------------------------------------------
# Shared helper: run bridge for all 4 steps, return split token groups
# ------------------------------------------------------------------
def _jepa_bridge_all_steps(batch):
    """Returns (all_steps_gated, vis_per_step, mot_per_step, text_per_step)."""
    hf   = batch["history_sig_first"].to("cuda", dtype=llm.dtype)
    hl   = batch["history_sig_last"].to("cuda",  dtype=llm.dtype)
    he   = batch["history_ego"].to("cuda", dtype=llm.dtype) if USE_MOTION else None

    all_steps_gated = []
    vis_per_step    = []
    mot_per_step    = []
    text_per_step   = []

    for t in range(4):
        sf  = hf[:, t];  sl = hl[:, t]
        ego = he[:, t] if he is not None else None

        # -- Vision tokens (split path for attribution) --
        vis_first = bridge.siglip_proj.float()(sf.float()).to(llm.dtype)
        vis_last  = bridge.siglip_proj.float()(sl.float()).to(llm.dtype)
        vis_last  = vis_last + bridge.vision_change_embed
        vis_tok   = torch.cat([vis_first, vis_last], dim=1)  # [B,1458,D]
        vis_tok   = modality_gate.gate_vision(vis_tok)

        # -- Motion tokens --
        mot_tok = None
        if USE_MOTION and ego is not None and bridge.use_motion:
            s, e = bridge._ego_last_patch_start, bridge._ego_last_patch_end
            k    = bridge.top_k_patches
            cls_raw      = ego[:, 0:1, :]
            mot_cls      = bridge.egovlp_proj.float()(cls_raw.float()).to(llm.dtype)
            mot_cls     += bridge.motion_cls_embed
            last_p       = ego[:, s:e, :].float()
            first_p      = ego[:, 1:197, :].float()
            norms        = last_p.norm(dim=-1)
            k_actual     = min(k, last_p.shape[1])
            topk_idx     = norms.topk(k_actual, dim=-1).indices
            idx_exp      = topk_idx.unsqueeze(-1).expand(-1, -1, ego.shape[-1])
            top_last_p   = torch.gather(last_p,  1, idx_exp)
            top_first_p  = torch.gather(first_p, 1, idx_exp)
            intra_raw    = (top_last_p - top_first_p).mean(1, keepdim=True)
            mot_intra    = bridge.egovlp_proj.float()(intra_raw.float()).to(llm.dtype)
            mot_intra   += bridge.motion_intra_delta_embed
            topk_proj    = bridge.egovlp_proj.float()(top_last_p.float()).to(llm.dtype)
            topk_proj   += bridge.motion_topk_rank_embeds.permute(1, 0, 2)
            mot_tok      = torch.cat([mot_cls, mot_intra, topk_proj], dim=1)
            gate_in, _   = bridge.cross_modal_gate(
                bridge.cross_modal_norm(mot_tok), vis_tok, vis_tok)
            mot_tok      = mot_tok + gate_in
            mot_tok      = modality_gate.gate_motion(mot_tok)
            # Align motion tokens into SigLIP subspace (matches training)
            mot_tok      = modality_align.align_motion(mot_tok)

        # -- Text tokens for this step --
        narration = batch["history_texts"][0][t]
        txt_ids   = tokenizer(narration, return_tensors="pt",
                            max_length=16, truncation=True,
                            padding="max_length").input_ids.to("cuda")
        with torch.no_grad():
            txt_tok = llm.base_model.model.model.embed_tokens(txt_ids)
        txt_tok = modality_gate.gate_text(txt_tok)
        # Align text tokens into SigLIP subspace (matches training)
        txt_tok = modality_align.align_text(txt_tok)

        # -- Fused step --
        fused = torch.cat([vis_tok, mot_tok], dim=1) if mot_tok is not None else vis_tok

        all_steps_gated.append(fused)
        vis_per_step.append(vis_tok)
        mot_per_step.append(mot_tok)
        text_per_step.append(txt_tok)

    return all_steps_gated, vis_per_step, mot_per_step, text_per_step

# ------------------------------------------------------------------
# Shared helper: context_encoder with attention capture
# ------------------------------------------------------------------
def _encode_with_attn(all_steps_gated, text_per_step):
    """
    Re-runs JEPAContextEncoder and captures cross-attention weights
    from each Perceiver layer.

    Returns (s_x, attn_maps)
    attn_maps: list of [B, H, num_latents, total_tokens] per layer
    """
    B = all_steps_gated[0].shape[0]

    # Stamp + fuse text (mirrors JEPAContextEncoder.forward)
    stamped = []
    for t_idx, step in enumerate(all_steps_gated):
        tok = step + context_encoder.time_embed[t_idx]
        txt = text_per_step[t_idx]
        txt = modality_gate.gate_text(txt)
        tok = torch.cat([tok, txt], dim=1)
        stamped.append(tok)

    # Change attention
    reference = stamped[0]
    changed   = [reference]
    for t_idx in range(1, len(stamped)):
        q_in  = context_encoder.change_norm(stamped[t_idx])
        delta, _ = context_encoder.change_attn(
            query=q_in, key=reference, value=reference)
        changed.append(stamped[t_idx] + delta)

    x       = torch.cat(changed, dim=1)
    latents = context_encoder.latents.unsqueeze(0).expand(B, -1, -1).clone()
    attn_maps = []

    for layer in context_encoder.layers:
        cross_in = layer["norm1"](latents)
        attn_out, attn_w = layer["cross_attn"](
            cross_in, x, x,
            need_weights=True,
            average_attn_weights=False   # [B, H, Q, K]
        )
        attn_maps.append(attn_w.detach().cpu().float())
        latents = latents + attn_out
        latents = latents + layer["ff"](latents)

    return context_encoder.norm(latents), attn_maps

# ==================================================================
# FIG 1  —  Context-encoder attention split by modality and timestep
# ==================================================================
def fig1_jepa_attention_split(batch, save_path):
    target = batch["target_text"][0]
    with torch.no_grad():
        all_steps, vis_per_step, mot_per_step, text_per_step = \
            _jepa_bridge_all_steps(batch)
        _, attn_maps = _encode_with_attn(all_steps, text_per_step)

    # Average over layers and heads → [K] weight per key token
    combined = torch.stack(
        [a[0].mean(dim=[0, 1]) for a in attn_maps], dim=0
    ).mean(0)
    combined = combined / combined.sum().clamp(min=1e-9)

    sig_weights  = np.zeros(4)
    ego_weights  = np.zeros(4)
    text_weights = np.zeros(4)
    ptr = 0
    for t in range(4):
        n_vis  = vis_per_step[t].shape[1]
        n_mot  = mot_per_step[t].shape[1] if mot_per_step[t] is not None else 0
        n_text = text_per_step[t].shape[1]
        n_total = n_vis + n_mot + n_text
        sig_weights[t]  = combined[ptr:ptr + n_vis].sum().item()
        ptr += n_vis
        if n_mot:
            ego_weights[t] = combined[ptr:ptr + n_mot].sum().item()
            ptr += n_mot
        text_weights[t] = combined[ptr:ptr + n_text].sum().item()
        ptr += n_text

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Fig 1 — Context-encoder attention by modality\nTarget: '{target}'",
                fontsize=9, ha="left", x=0.02)

    x = np.arange(4); w = 0.25
    labels_t = ["t-4", "t-3", "t-2", "t-1"]
    axes[0].bar(x - w,   sig_weights  * 100, w, label="SigLIP (vision)",  color="#4C8EDA")
    if USE_MOTION:
        axes[0].bar(x,   ego_weights  * 100, w, label="EgoVLP (motion)",  color="#E07B54")
    axes[0].bar(x + w,   text_weights * 100, w, label="Text history",     color="#9B59B6")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels_t)
    axes[0].set_ylabel("% of total attention"); axes[0].set_ylim(0, 55)
    axes[0].set_title("Attention share per timestep"); axes[0].legend(fontsize=8)

    sizes  = [sig_weights.sum(), text_weights.sum()]
    labels = ["SigLIP vision", "Text history"]
    colors = ["#4C8EDA", "#9B59B6"]
    if USE_MOTION:
        sizes.insert(1, ego_weights.sum())
        labels.insert(1, "EgoVLP motion")
        colors.insert(1, "#E07B54")
    axes[1].pie(sizes, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    axes[1].set_title("Overall modality attention split")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save_path}")

# ==================================================================
# FIG 2  —  Per-latent temporal attention heatmap
# ==================================================================
def fig2_jepa_temporal_heatmap(batch, save_path):
    target = batch["target_text"][0]
    with torch.no_grad():
        all_steps, vis_per_step, mot_per_step, text_per_step = \
            _jepa_bridge_all_steps(batch)
        _, attn_maps = _encode_with_attn(all_steps, text_per_step)

    avg_attn = torch.stack(
        [a[0].mean(dim=0) for a in attn_maps], dim=0
    ).mean(0).numpy()                    # [num_latents, K]

    num_latents = avg_attn.shape[0]
    # Tokens per timestep = n_vis + n_mot + n_text
    tps = []
    for t in range(4):
        n  = vis_per_step[t].shape[1]
        n += mot_per_step[t].shape[1] if mot_per_step[t] is not None else 0
        n += text_per_step[t].shape[1]
        tps.append(n)

    per_timestep = np.zeros((num_latents, 4))
    ptr = 0
    for t, n in enumerate(tps):
        per_timestep[:, t] = avg_attn[:, ptr:ptr + n].sum(-1)
        ptr += n
    row_max = per_timestep.max(1, keepdims=True).clip(min=1e-9)
    per_timestep /= row_max

    fig, ax = plt.subplots(figsize=(7, max(4, num_latents // 20)))
    im = ax.imshow(per_timestep, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Timestep"); ax.set_ylabel(f"Latent (0–{num_latents-1})")
    ax.set_xticks([0,1,2,3]); ax.set_xticklabels(["t-4","t-3","t-2","t-1"])
    ax.set_title(f"Fig 2 — Per-latent temporal attention\nTarget: '{target}'", fontsize=9)
    plt.colorbar(im, ax=ax, label="Relative attention (row-normalised)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save_path}")

# ==================================================================
# FIG 3  —  Modality ablation (confidence drop when each is zeroed)
# ==================================================================
def fig3_jepa_ablation(batch, save_path):
    target_text = batch["target_text"][0]
    prompt_text = batch["prompt_text"][0]
    target_ids  = tokenizer(target_text, return_tensors="pt").input_ids.to("cuda")
    prompt_ids  = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")

    def _logprob(zero_vis=False, zero_mot=False, zero_text=False,
                alt_prompt_ids=None):
        with torch.no_grad():
            hf  = batch["history_sig_first"].to("cuda", dtype=llm.dtype)
            hl  = batch["history_sig_last"].to("cuda",  dtype=llm.dtype)
            he  = batch["history_ego"].to("cuda", dtype=llm.dtype) if USE_MOTION else None
            all_steps_abl = []
            for t in range(4):
                sf  = hf[:, t]; sl = hl[:, t]
                ego = he[:, t] if he is not None else None
                vis_f = bridge.siglip_proj.float()(sf.float()).to(llm.dtype)
                vis_l = bridge.siglip_proj.float()(sl.float()).to(llm.dtype)
                vis_l = vis_l + bridge.vision_change_embed
                vis   = torch.cat([vis_f, vis_l], dim=1)
                vis   = modality_gate.gate_vision(vis)
                if zero_vis:
                    vis = torch.zeros_like(vis)

                mot = None
                if USE_MOTION and ego is not None and bridge.use_motion:
                    s, e  = bridge._ego_last_patch_start, bridge._ego_last_patch_end
                    k     = bridge.top_k_patches
                    c_raw = ego[:, 0:1, :]
                    mc    = bridge.egovlp_proj.float()(c_raw.float()).to(llm.dtype)
                    mc   += bridge.motion_cls_embed
                    lp    = ego[:, s:e, :].float()
                    fp    = ego[:, 1:197, :].float()
                    nrms  = lp.norm(dim=-1)
                    ki    = nrms.topk(min(k, lp.shape[1]), dim=-1).indices
                    ie    = ki.unsqueeze(-1).expand(-1, -1, ego.shape[-1])
                    tlp   = torch.gather(lp, 1, ie)
                    tfp   = torch.gather(fp, 1, ie)
                    intra = bridge.egovlp_proj.float()(
                        (tlp-tfp).mean(1,keepdim=True).float()).to(llm.dtype)
                    intra += bridge.motion_intra_delta_embed
                    tkp   = bridge.egovlp_proj.float()(tlp.float()).to(llm.dtype)
                    tkp  += bridge.motion_topk_rank_embeds.permute(1, 0, 2)
                    mot   = torch.cat([mc, intra, tkp], dim=1)
                    gv, _ = bridge.cross_modal_gate(
                        bridge.cross_modal_norm(mot), vis, vis)
                    mot   = modality_gate.gate_motion(mot + gv)
                    if zero_mot:
                        mot = torch.zeros_like(mot)

                narration = batch["history_texts"][0][t]
                txt_ids_t = tokenizer(narration, return_tensors="pt",
                                    max_length=16, truncation=True,
                                    padding="max_length").input_ids.to("cuda")
                txt = llm.base_model.model.model.embed_tokens(txt_ids_t)
                txt = modality_gate.gate_text(txt)
                if zero_text:
                    txt = torch.zeros_like(txt)

                fused = torch.cat([vis, mot], dim=1) if mot is not None else vis
                all_steps_abl.append(fused)
                # text_per_step built inline below for encoder call
            txt_list = []
            for t in range(4):
                narration = batch["history_texts"][0][t]
                tid = tokenizer(narration, return_tensors="pt",
                                max_length=16, truncation=True,
                                padding="max_length").input_ids.to("cuda")
                te  = llm.base_model.model.model.embed_tokens(tid)
                te  = modality_gate.gate_text(te)
                if zero_text:
                    te = torch.zeros_like(te)
                txt_list.append(te)

            s_x   = context_encoder(all_steps_abl, txt_list, modality_gate)
            s_hat = predictor(s_x)

            use_ids = alt_prompt_ids if alt_prompt_ids is not None else prompt_ids
            p_emb   = llm.base_model.model.model.embed_tokens(use_ids)
            t_emb   = llm.base_model.model.model.embed_tokens(target_ids)
            full    = torch.cat([s_hat, p_emb, t_emb], dim=1)
            ignore  = s_hat.shape[1] + p_emb.shape[1]
            lbl     = torch.full((1, full.shape[1]), -100, dtype=torch.long, device="cuda")
            lbl[0, ignore:] = target_ids[0]
            return -llm(inputs_embeds=full, labels=lbl).loss.item()

    masked_prompt = (prompt_text.split("Previous actions:")[0] +
                    "Previous actions: None available. Predict based on video.\n\n"
                    "Predict the next action.:<|im_end|>\n<|im_start|>assistant\n")
    masked_ids = tokenizer(masked_prompt, return_tensors="pt").input_ids.to("cuda")

    lp_full     = _logprob()
    lp_no_vis   = _logprob(zero_vis=True)
    lp_no_mot   = _logprob(zero_mot=True)  if USE_MOTION else lp_full
    lp_no_both  = _logprob(zero_vis=True, zero_mot=True)
    lp_no_text  = _logprob(alt_prompt_ids=masked_ids)

    conds   = ["Full model", "Vision\nzeroed", "Motion\nzeroed",
            "All visual\nzeroed", "Text hist\nmasked"]
    lps     = [lp_full, lp_no_vis, lp_no_mot, lp_no_both, lp_no_text]
    drops   = [lp_full - lp for lp in lps]
    colors  = ["#5BAD72", "#4C8EDA", "#E07B54", "#2E86AB", "#9B59B6"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Fig 3 — Modality ablation\nTarget: '{target_text}'",
                fontsize=9, ha="left", x=0.02)
    axes[0].bar(conds, lps,   color=colors, alpha=0.85)
    axes[0].axhline(lp_full, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Log-probability (higher = more confident)")
    axes[0].set_title("Absolute confidence per condition")
    for i, v in enumerate(lps):
        axes[0].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axes[1].bar(conds, drops, color=colors, alpha=0.85)
    axes[1].set_ylabel("Confidence drop (full − ablated)")
    axes[1].set_title("Modality importance")
    for i, v in enumerate(drops):
        axes[1].text(i, max(v, 0) + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save_path}")

# ==================================================================
# FIG 4  —  PCA of context-encoder input tokens by modality/timestep
# ==================================================================
def fig4_jepa_pca(batch, save_path):
    target = batch["target_text"][0]
    with torch.no_grad():
        _, vis_per_step, mot_per_step, text_per_step = \
            _jepa_bridge_all_steps(batch)

    all_tokens  = []
    labels_mod  = []
    labels_time = []
    for t in range(4):
        v = vis_per_step[t][0].float().cpu().numpy()
        all_tokens.append(v)
        labels_mod  += ["SIG"]  * v.shape[0]
        labels_time += [t]      * v.shape[0]
        if mot_per_step[t] is not None:
            m = mot_per_step[t][0].float().cpu().numpy()
            all_tokens.append(m)
            labels_mod  += ["EGO"]  * m.shape[0]
            labels_time += [t]      * m.shape[0]
        tx = text_per_step[t][0].float().cpu().numpy()
        all_tokens.append(tx)
        labels_mod  += ["TXT"]  * tx.shape[0]
        labels_time += [t]      * tx.shape[0]

    X   = np.concatenate(all_tokens, axis=0)
    pca = PCA(n_components=2, random_state=0)
    X2  = pca.fit_transform(X)
    lm  = np.array(labels_mod); lt = np.array(labels_time)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Fig 4 — PCA of context-encoder input tokens\nTarget: '{target}'",
                fontsize=9)
    for mod, col, marker, size in [
        ("SIG", "#4C8EDA", ".", 1),
        ("EGO", "#E07B54", "*", 20),
        ("TXT", "#9B59B6", "D", 10),
    ]:
        mask = lm == mod
        if mask.any():
            axes[0].scatter(X2[mask, 0], X2[mask, 1], s=size,
                            alpha=0.3 if mod == "SIG" else 0.9,
                            color=col, marker=marker,
                            label=f"{mod} ({mask.sum()})", zorder=3 if mod != "SIG" else 1)
    axes[0].set_title("Token space by modality")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].legend(markerscale=4, fontsize=8)

    cmap = plt.cm.plasma
    for t in range(4):
        mask = lt == t
        axes[1].scatter(X2[mask & (lm == "SIG"), 0],
                        X2[mask & (lm == "SIG"), 1],
                        s=1, alpha=0.2, color=cmap(t / 3), label=f"t-{4-t}")
        for mod, marker, size in [("EGO", "*", 30), ("TXT", "D", 15)]:
            mm = mask & (lm == mod)
            if mm.any():
                axes[1].scatter(X2[mm, 0], X2[mm, 1], s=size, alpha=1.0,
                                color=cmap(t / 3), marker=marker, zorder=5)
    axes[1].set_title("Token space by timestep (★=EGO ◆=TXT)")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].legend(markerscale=4, fontsize=8, title="Timestep")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save_path}")

# ==================================================================
# FIG 5  —  Text token saliency (gradient × embedding)
# ==================================================================
def fig5_jepa_text_saliency(batch, save_path):
    target_text = batch["target_text"][0]
    prompt_text = batch["prompt_text"][0]
    target_ids  = tokenizer(target_text, return_tensors="pt").input_ids.to("cuda")
    prompt_ids  = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")

    MAX_SHOW = 40
    raw_labels = [tokenizer.decode([i]).replace("Ġ", " ") for i in prompt_ids[0]]
    prompt_ids_grad = prompt_ids[:, -MAX_SHOW:] if len(raw_labels) > MAX_SHOW \
                    else prompt_ids
    if len(raw_labels) > MAX_SHOW:
        raw_labels = ["..."] + raw_labels[-MAX_SHOW:]

    with torch.no_grad():
        all_steps_g, _, _, text_per_step = _jepa_bridge_all_steps(batch)
        s_x   = context_encoder(all_steps_g, text_per_step, modality_gate)
        s_hat = predictor(s_x)

    p_emb = llm.base_model.model.model.embed_tokens(
        prompt_ids_grad).detach().requires_grad_(True)
    t_emb = llm.base_model.model.model.embed_tokens(target_ids).detach()

    full   = torch.cat([s_hat.detach(), p_emb, t_emb], dim=1)
    ignore = s_hat.shape[1] + p_emb.shape[1]
    labels = torch.full((1, full.shape[1]), -100, dtype=torch.long, device="cuda")
    labels[0, ignore:] = target_ids[0]
    (-llm(inputs_embeds=full, labels=labels).loss).backward()

    saliency = (p_emb.grad * p_emb).abs().sum(-1)[0].float().cpu().numpy()
    if len(saliency) > MAX_SHOW:
        saliency = saliency[-MAX_SHOW:]
    saliency = saliency / saliency.max().clip(min=1e-9)

    fig, ax = plt.subplots(figsize=(max(10, len(raw_labels) * 0.4), 3.5))
    ax.bar(range(len(raw_labels)), saliency,
        color=plt.cm.Reds(saliency), edgecolor="none")
    ax.set_xticks(range(len(raw_labels)))
    ax.set_xticklabels(raw_labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Gradient saliency (normalised)")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"Fig 5 — Text token saliency (gradient × embedding)\n"
        f"Target: '{target_text}'  |  Last {MAX_SHOW} prompt tokens",
        fontsize=9
    )
    sm = ScalarMappable(cmap="Reds", norm=Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Saliency", shrink=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save_path}")

# ==================================================================
# FIG 6  —  JEPA representation alignment  (NEW — no equivalent in lora_47)
#
# For each sample plots cosine similarity between:
#   s_x (context latent)  vs  s_ŷ (predictor output)
#   s_x                   vs  s_y (EMA target)
#   s_ŷ                   vs  s_y
#
# A well-trained JEPA predictor should show:
#   sim(s_hat, s_y)  close to  sim(s_x, s_y)  or higher
#   sim(s_hat, s_y)  >> sim(s_x, s_y)  means predictor adds value
# Also plots per-latent similarity heatmap to spot dead latents.
# ==================================================================
def fig6_jepa_representation_alignment(batch, save_path):
    target_text = batch["target_text"][0]

    hf  = batch["history_sig_first"].to("cuda", dtype=llm.dtype)
    hl  = batch["history_sig_last"].to("cuda",  dtype=llm.dtype)
    he  = batch["history_ego"].to("cuda", dtype=llm.dtype) if USE_MOTION else None
    t0_sf = batch["t0_sig_first"].to("cuda", dtype=llm.dtype)
    t0_sl = batch["t0_sig_last"].to("cuda",  dtype=llm.dtype)
    t0_eg = batch["t0_ego"].to("cuda", dtype=llm.dtype) if USE_MOTION else None

    with torch.no_grad():
        # Context path
        all_steps_g, _, _, text_per_step = _jepa_bridge_all_steps(batch)
        s_x   = context_encoder(all_steps_g, text_per_step, modality_gate)
        s_hat = predictor(s_x)

        # Target path (t-0 features through target encoder)
        t0_vis = bridge(t0_sf, t0_sl, t0_eg if USE_MOTION else None)
        t0_vis = modality_gate.gate_vision(t0_vis)
        t0_txt_ids = tokenizer(
            target_text, return_tensors="pt",
            max_length=16, truncation=True, padding="max_length"
        ).input_ids.to("cuda")
        t0_txt = llm.base_model.model.model.embed_tokens(t0_txt_ids)
        s_y = target_encoder.encode_target([t0_vis]*4, [t0_txt]*4, modality_gate)

        # Normalise all three to unit sphere for cosine similarity
        sx_n   = torch.nn.functional.normalize(s_x.float(),   dim=-1)[0]  # [L, D]
        shat_n = torch.nn.functional.normalize(s_hat.float(),  dim=-1)[0]
        sy_n   = torch.nn.functional.normalize(s_y.float(),    dim=-1)[0]

        # Per-latent cosine similarity (dot product of unit vectors)
        sim_sx_shat = (sx_n   * shat_n).sum(-1).cpu().numpy()   # [L]
        sim_sx_sy   = (sx_n   * sy_n  ).sum(-1).cpu().numpy()
        sim_shat_sy = (shat_n * sy_n  ).sum(-1).cpu().numpy()

    L = len(sim_sx_shat)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        f"Fig 6 — JEPA representation alignment\nTarget: '{target_text}'",
        fontsize=9, ha="left", x=0.02
    )

    # Left: per-latent cosine-sim curves
    x_ax = np.arange(L)
    axes[0].plot(x_ax, sim_sx_shat,  color="#4C8EDA", lw=0.8, label="s_x  vs  s_ŷ (predictor in)")
    axes[0].plot(x_ax, sim_sx_sy,    color="#9B59B6", lw=0.8, label="s_x  vs  s_y (target)")
    axes[0].plot(x_ax, sim_shat_sy,  color="#E07B54", lw=1.2, label="s_ŷ  vs  s_y  ← key signal")
    axes[0].axhline(0, color="gray", lw=0.5, linestyle="--")
    axes[0].set_xlabel(f"Latent index (0–{L-1})")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_ylim(-0.2, 1.05)
    axes[0].set_title("Per-latent cosine similarity")
    axes[0].legend(fontsize=8)

    # Right: mean summary bars
    means  = [sim_sx_shat.mean(), sim_sx_sy.mean(), sim_shat_sy.mean()]
    stds   = [sim_sx_shat.std(),  sim_sx_sy.std(),  sim_shat_sy.std() ]
    b_lbls = ["s_x vs s_ŷ", "s_x vs s_y", "s_ŷ vs s_y"]
    b_cols = ["#4C8EDA", "#9B59B6", "#E07B54"]
    bars = axes[1].bar(b_lbls, means, yerr=stds, color=b_cols,
                    alpha=0.85, capsize=4)
    axes[1].set_ylabel("Mean cosine similarity (± std)")
    axes[1].set_ylim(-0.1, 1.05)
    axes[1].set_title("Summary — higher s_ŷ vs s_y = better predictor")
    for bar, v, s in zip(bars, means, stds):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                    v + s + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save_path}")
    
#########################################################################################################
# ===========================================================================
# 12. MAIN
# ===========================================================================
if __name__ == '__main__':

    current_file = os.path.basename(__file__).split(".")[0]

    EXTRACT_FEATURES          = False
    LOAD_PRETRAINED_BRIDGE    = False
    USE_PREEXTRACTED_FEATURES = True

    USE_MOTION                = True

    TRAIN_BRIDGE              = True
    VALIDATION_BRIDGE         = False
    
    INFERENCE_BRIDGE          = True
    ANALYSE_INFERENCE         = True
    VISUALIZE_FRAMES          = True 

    # JEPA-specific hyperparameters
    EMA_TAU            = 0.996      # target encoder update rate
    JEPA_LOSS_WEIGHT   = 1.0        # weight on the JEPA latent L2 loss
    LM_LOSS_WEIGHT     = 1.0        # weight on the language model CE loss
    # Fraction of t-0 vision tokens zeroed before the target encoder.
    # Ensures s_y != s_x even when EMA weights are close to context_encoder.
    # 0.15 = 15% dropout.  Range: 0.10–0.25.  Set to 0.0 to disable.
    TARGET_TOKEN_DROPOUT = 0.15

    print(f"USE_MOTION: {USE_MOTION} | TRAIN: {TRAIN_BRIDGE}")

    set_seed(42)
    os.makedirs("checkpoints", exist_ok=True)

    # --- LLM ---
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa"
    )
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False
    llm.config.use_cache = False

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()
    llm.train()
    llm.config.use_cache = False

    # --- SigLIP ---
    vision_model = SiglipVisionModel.from_pretrained(
        "google/siglip-so400m-patch14-384", torch_dtype=torch.float16).to("cuda")
    vision_model.eval()
    print("Loaded SigLIP")

    # --- EgoVLP ---
    print("Loading EgoVLP Motion Encoder...")
    EGO_REPO_PATH = os.path.abspath(os.path.join(BASE_DIR, "EgoVLPv2", "EgoVLPv2"))
    if EGO_REPO_PATH not in sys.path:
        sys.path.insert(0, EGO_REPO_PATH)
    original_cwd = os.getcwd()
    os.chdir(EGO_REPO_PATH)
    try:
        from model.roberta import RobertaModel
        if not hasattr(RobertaModel, 'all_tied_weights_keys'):
            RobertaModel.all_tied_weights_keys = property(lambda self: {})
        from model.model import FrozenInTime as EgoModel
    except ImportError as e:
        print(f"Import failed: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_cwd)

    ego_config = {
        'video_params': {'model': 'SpaceTimeTransformer',
                         'arch_config': 'base_patch16_224',
                         'num_frames': 4, 'pretrained': True},
        'text_params':  {'model': 'roberta-base', 'pretrained': True},
        'projection_dim': 4096,
    }
    CHECKPOINT_PATH = r"C:\MultiModal\multiModel\pretrain_model\EgoVLPv2.pth"
    motion_model = load_egovlp(CHECKPOINT_PATH)
    motion_model.eval()
    print("Loaded EgoVLPv2")

    # --- Dimensions ---
    llm_dim    = llm.config.hidden_size    # 3584
    siglip_dim = vision_model.config.hidden_size  # 1152
    egovlp_dim = 768

    print(f"{'LLM dim':<20} | {llm_dim}")
    print(f"{'SigLIP dim':<20} | {siglip_dim}")
    print(f"{'EgoVLP dim':<20} | {egovlp_dim}")

    # --- Build JEPA modules ---
    modality_gate = ModalityGate(
        init_logit_vision=5.0,   # α_vision ≈ 1.0 (open)
        init_logit_motion=5.0,   # α_motion ≈ 1.0 (open)
        init_logit_text=-5.0     # α_text   ≈ 0.0 (closed → opens during training)
    ).to("cuda").to(llm.dtype)

    modality_align = ModalityAlignmentProjector(
        dim=llm_dim,
    ).to("cuda").to(llm.dtype)

    bridge = MultimodalBridge(
        siglip_dim=siglip_dim,
        egovlp_dim=egovlp_dim,
        llm_dim=llm_dim,
        use_motion=USE_MOTION,
        top_k_patches=8,
    ).to("cuda").to(llm.dtype)

    context_encoder = JEPAContextEncoder(
        dim=llm_dim,
        num_latents=128,
        depth=2,
        num_heads=8,
        num_steps=4,
    ).to("cuda").to(llm.dtype)

    target_encoder = EMATargetEncoder(
        online_encoder=context_encoder,
        tau=EMA_TAU,
    ).to("cuda")   # params already non-requiring-grad
    # Cast EMA encoder to the right dtype
    target_encoder.encoder.to(llm.dtype)

    predictor = JEPAPredictor(
        dim=llm_dim,
        predictor_dim=512,       # bottleneck width
        num_latents=128,
        depth=4,
        num_heads=8,
    ).to("cuda").to(llm.dtype)

    # --- Optimiser param groups ---
    if USE_MOTION:
        motion_param_ids = set(
            id(p) for p in (
                list(bridge.egovlp_proj.parameters()) +
                [bridge.motion_cls_embed,
                 bridge.motion_intra_delta_embed,
                 bridge.motion_topk_rank_embeds] +
                list(bridge.cross_modal_gate.parameters()) +
                list(bridge.cross_modal_norm.parameters())
            ))
        spatial_bridge_params = [p for p in bridge.parameters() if id(p) not in motion_param_ids]
        motion_bridge_params  = [p for p in bridge.parameters() if id(p) in motion_param_ids]
        bridge_groups = [
            {'params': spatial_bridge_params,            'lr': 1e-4},
            {'params': motion_bridge_params,             'lr': 2e-5},
        ]
    else:
        bridge_groups = [{'params': bridge.parameters(), 'lr': 1e-4}]

    optimizer = torch.optim.AdamW(
        bridge_groups + [
            {'params': context_encoder.parameters(),                    'lr': 1e-4},
            {'params': predictor.parameters(),                          'lr': 1e-4},
            {'params': modality_gate.parameters(),                      'lr': 5e-4},
            {'params': modality_align.parameters(),                     'lr': 1e-4},
            {'params': filter(lambda p: p.requires_grad, llm.parameters()), 'lr': 2e-5},
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # --- Transforms ---
    siglip_transform = T.Compose([
        T.ToPILImage(), T.Resize((384, 384)), T.ToTensor()])
    egovlp_transform = T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.ToTensor()])

    # --- Dataset ---
    CACHE_DIRECTORY = "extracted_features_new"
    dataset = EpicKitchensDataset(
        csv_path="EPIC_KITCHENS_P01_captions.csv",
        siglip_transform=siglip_transform,
        egovlp_transform=egovlp_transform,
        history_len=4,
        tokenizer=tokenizer,
        cache_dir=CACHE_DIRECTORY,
    )
    dataset_size = len(dataset)
    train_split  = 0.1
    train_size   = int(train_split * dataset_size)
    val_size     = int(0.8 * dataset_size)
    test_size    = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42))

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    num_workers  = 1
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False, prefetch_factor=4, collate_fn=collate_fn)
    val_loader   = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False, collate_fn=collate_fn)
    test_loader  = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False, collate_fn=collate_fn)

    # --- Feature extraction ---
    if EXTRACT_FEATURES:
        extract_dataset = EpicKitchensDataset(
            csv_path="EPIC_KITCHENS_P01_captions.csv",
            siglip_transform=siglip_transform,
            egovlp_transform=egovlp_transform,
            history_len=4, tokenizer=tokenizer,
            cache_dir=CACHE_DIRECTORY, extraction_mode=True,
        )
        extract_loader = DataLoader(
            extract_dataset, batch_size=4, shuffle=False,
            num_workers=0, collate_fn=collate_fn)
        extract_and_save_features(
            dataloader=extract_loader,
            output_dir=CACHE_DIRECTORY,
            vision_model=vision_model,
            motion_model=motion_model,
        )
        print("Extraction done. Set EXTRACT_FEATURES=False and re-run.")
        sys.exit(0)

    if USE_PREEXTRACTED_FEATURES:
        print("Releasing vision/motion encoders from VRAM...")
        vision_model.cpu(); motion_model.cpu()
        del vision_model, motion_model
        gc.collect(); torch.cuda.empty_cache()
        print(f"VRAM after release: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    if TRAIN_BRIDGE:
        print("-" * 60)
        print(f"Bridge params       : {count_parameters(bridge):,}")
        print(f"Context encoder     : {count_parameters(context_encoder):,}")
        print(f"Predictor           : {count_parameters(predictor):,}")
        print(f"Modality gate       : {count_parameters(modality_gate):,}")
        print(f"LLM (LoRA)          : {count_parameters(llm):,}")
        print("-" * 60)

        total_epochs = 8
        # Text gate warm-up schedule:
        # epoch 0-1: α_text stays near 0 (gate logit clamped low)
        # epoch 2+:  gate logit is free to learn
        TEXT_GATE_WARMUP_EPOCHS = 2

        for epoch_full in range(total_epochs):

            bridge.train()
            context_encoder.train()
            predictor.train()
            modality_gate.train()
            llm.train()

            # Freeze text gate for first TEXT_GATE_WARMUP_EPOCHS epochs
            freeze_text_gate = epoch_full < TEXT_GATE_WARMUP_EPOCHS
            modality_gate.logit_text.requires_grad_(not freeze_text_gate)

            epoch_loss_sum  = 0.0
            epoch_lm_sum    = 0.0
            epoch_jepa_sum  = 0.0
            vis_var_sum     = 0.0
            num_batches     = 0
            LOG_EVERY       = 50

            pbar = tqdm(train_loader,
                        desc=f"Epoch {epoch_full+1}/{total_epochs}")

            for batch in pbar:
                if batch is None:
                    continue

                optimizer.zero_grad()

                hist_sf  = batch["history_sig_first"].to("cuda", dtype=llm.dtype)
                hist_sl  = batch["history_sig_last"].to("cuda",  dtype=llm.dtype)
                hist_ego = batch["history_ego"].to("cuda",       dtype=llm.dtype)
                t0_sf    = batch["t0_sig_first"].to("cuda",      dtype=llm.dtype)
                t0_sl    = batch["t0_sig_last"].to("cuda",       dtype=llm.dtype)
                t0_ego   = batch["t0_ego"].to("cuda",            dtype=llm.dtype)

                # --- Build per-step bridge outputs with modality alignment ---
                # bridge() returns vis_tokens + mot_tokens concatenated.
                # We split them, align each non-vision modality into the SigLIP
                # subspace via modality_align, then re-concatenate.
                all_visual_steps = []
                for t_idx in range(4):
                    sf   = hist_sf[:,  t_idx, :, :]
                    sl   = hist_sl[:,  t_idx, :, :]
                    ego  = hist_ego[:, t_idx, :, :] if USE_MOTION else None

                    if USE_MOTION and ego is not None:
                        # Run bridge internals split so we can align mot separately
                        vis_f = bridge.siglip_proj.float()(sf.float()).to(llm.dtype)
                        vis_l = bridge.siglip_proj.float()(sl.float()).to(llm.dtype)
                        vis_l = vis_l + bridge.vision_change_embed
                        vis_tok = torch.cat([vis_f, vis_l], dim=1)   # [B,1458,D]
                        vis_tok = modality_gate.gate_vision(vis_tok)

                        # Reconstruct motion tokens (same logic as bridge.forward)
                        s, e = bridge._ego_last_patch_start, bridge._ego_last_patch_end
                        k    = bridge.top_k_patches
                        cls_raw = ego[:, 0:1, :]
                        mot_cls = bridge.egovlp_proj.float()(cls_raw.float()).to(llm.dtype)
                        mot_cls += bridge.motion_cls_embed
                        lp = ego[:, s:e, :].float(); fp = ego[:, 1:197, :].float()
                        nrms = lp.norm(dim=-1)
                        ki   = nrms.topk(min(k, lp.shape[1]), dim=-1).indices
                        ie   = ki.unsqueeze(-1).expand(-1, -1, ego.shape[-1])
                        tlp  = torch.gather(lp, 1, ie); tfp = torch.gather(fp, 1, ie)
                        intra = bridge.egovlp_proj.float()(
                            (tlp-tfp).mean(1,keepdim=True).float()).to(llm.dtype)
                        intra += bridge.motion_intra_delta_embed
                        tkp   = bridge.egovlp_proj.float()(tlp.float()).to(llm.dtype)
                        tkp  += bridge.motion_topk_rank_embeds.permute(1, 0, 2)
                        mot_tok = torch.cat([mot_cls, intra, tkp], dim=1)
                        gate_in = bridge.cross_modal_norm(mot_tok)
                        gated, _ = bridge.cross_modal_gate(
                            query=gate_in, key=vis_tok, value=vis_tok)
                        mot_tok = mot_tok + gated
                        mot_tok = modality_gate.gate_motion(mot_tok)

                        # Align motion tokens into the SigLIP subspace
                        mot_tok = modality_align.align_motion(mot_tok)

                        vis = torch.cat([vis_tok, mot_tok], dim=1)
                    else:
                        vis = bridge(sf, sl, None)
                        vis = modality_gate.gate_vision(vis)

                    all_visual_steps.append(vis)

                # Build text token embeddings — aligned into SigLIP subspace
                with torch.no_grad():
                    text_embeds_list = []
                    for t_idx in range(4):
                        narration  = batch["history_texts"][0][t_idx]
                        txt_ids    = tokenizer(narration, return_tensors="pt",
                                               max_length=16, truncation=True,
                                               padding="max_length").input_ids.to("cuda")
                        txt_embed  = llm.base_model.model.model.embed_tokens(txt_ids)
                        text_embeds_list.append(txt_embed)  # [B, T_txt, D]
                # Align text embeddings (done outside no_grad so alignment MLP trains)
                text_embeds_list = [modality_align.align_text(t) for t in text_embeds_list]

                # Build t-0 context for target encoder (FIXED).
                #
                # Root cause of s_x vs s_y = 1.000 in lora_47/48:
                #   [t0_vis]*4 fed the same t-0 tensor to all 4 time slots.
                #   change-attention deltas were all zero, the target encoder
                #   (an EMA copy started from context_encoder weights) produced
                #   an output numerically identical to what context_encoder
                #   would produce on the same input → cos_sim = 1.0.
                #
                # Two-part fix:
                #   1. Pass [t0_vis] as a single-element list.
                #      len(stamped)=1 so the change-attn loop body never runs
                #      (range(1,1) is empty).  The encoder still runs its full
                #      cross-attention stack on the t-0 tokens, producing a
                #      genuine t-0 representation that the predictor must learn
                #      to predict from the t-4…t-1 context.
                #   2. Token dropout (TARGET_TOKEN_DROPOUT, default 15%):
                #      randomly zero vision tokens before the target encoder so
                #      that s_y carries a corrupted-but-still-informative view
                #      of t-0.  This breaks any trivial s_y ≈ s_x path that
                #      might survive the single-step fix (e.g. if the EMA
                #      encoder converges to the identity on this distribution).
                with torch.no_grad():
                    t0_vis  = bridge(t0_sf, t0_sl, t0_ego if USE_MOTION else None)
                    t0_vis  = modality_gate.gate_vision(t0_vis)

                    # Token dropout: zero TOKEN_DROPOUT_RATE of vision tokens
                    # This must only happen during training, not eval/inference.
                    dropout_mask = (
                        torch.rand(t0_vis.shape[:2], device=t0_vis.device)
                        > TARGET_TOKEN_DROPOUT
                    ).unsqueeze(-1).to(t0_vis.dtype)
                    t0_vis_dropped = t0_vis * dropout_mask   # [B, N_vis, D]

                    t0_txt_ids = tokenizer(
                        batch["target_text"][0], return_tensors="pt",
                        max_length=16, truncation=True,
                        padding="max_length").input_ids.to("cuda")
                    t0_txt = modality_align.align_text(
                        llm.base_model.model.model.embed_tokens(t0_txt_ids))

                    # Single-step list: change-attn loop is skipped (range(1,1))
                    # but the full Perceiver cross-attention still runs.
                    s_y = target_encoder.encode_target(
                        [t0_vis_dropped], [t0_txt], modality_gate)

                # --- Online context encoder → s_x ---
                s_x = context_encoder(all_visual_steps, text_embeds_list, modality_gate)

                # --- Predictor → s_hat ---
                s_hat = predictor(s_x)

                # Collapse check
                vis_var = s_hat[0].float().var(dim=0).mean().item()
                vis_var_sum += vis_var
                if vis_var < 1e-5 and num_batches > 10:
                    print(f"\n[COLLAPSE] vis_var={vis_var:.2e} at batch {num_batches}. "
                          f"Stopping epoch early.")
                    break

                # --- JEPA loss ---
                j_loss = jepa_loss(s_hat, s_y, normalize=True)

                # --- LM loss: decode from s_hat ---
                with torch.no_grad():
                    p_embeds = llm.base_model.model.model.embed_tokens(
                        tokenizer(batch["prompt_text"][0], return_tensors="pt"
                                  ).input_ids.to("cuda"))
                    t_embeds = llm.base_model.model.model.embed_tokens(
                        tokenizer(batch["target_text"][0], return_tensors="pt"
                                  ).input_ids.to("cuda"))

                full_embeds = torch.cat([s_hat, p_embeds, t_embeds], dim=1)
                ignore_len  = s_hat.shape[1] + p_embeds.shape[1]
                labels      = torch.full((1, full_embeds.shape[1]), -100,
                                         dtype=torch.long, device="cuda")
                labels[0, ignore_len:] = tokenizer(
                    batch["target_text"][0], return_tensors="pt").input_ids[0].to("cuda")

                lm_out   = llm(inputs_embeds=full_embeds, labels=labels)
                lm_loss  = lm_out.loss

                # --- Combined loss ---
                loss = JEPA_LOSS_WEIGHT * j_loss + LM_LOSS_WEIGHT * lm_loss
                loss.backward()

                all_params = (
                    list(bridge.parameters()) +
                    list(context_encoder.parameters()) +
                    list(predictor.parameters()) +
                    list(modality_gate.parameters()) +
                    list(modality_align.parameters()) +
                    [p for p in llm.parameters() if p.requires_grad]
                )
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

                # EMA update of target encoder AFTER the main backward
                target_encoder.update(context_encoder)

                epoch_loss_sum  += loss.item()
                epoch_lm_sum    += lm_loss.item()
                epoch_jepa_sum  += j_loss.item()
                num_batches     += 1

                pbar.set_postfix(
                    jepa=f"{j_loss.item():.3f}",
                    lm=f"{lm_loss.item():.3f}",
                    α_txt=f"{modality_gate.alpha_text.item():.3f}",
                    α_vis=f"{modality_gate.alpha_vision.item():.3f}",
                )

                if num_batches % LOG_EVERY == 0:
                    print(
                        f"\n  [b{num_batches}] "
                        f"jepa={epoch_jepa_sum/num_batches:.4f} | "
                        f"lm={epoch_lm_sum/num_batches:.4f} | "
                        f"vis_var={vis_var_sum/num_batches:.6f} | "
                        f"α_vis={modality_gate.alpha_vision.item():.3f} | "
                        f"α_mot={modality_gate.alpha_motion.item():.3f} | "
                        f"α_txt={modality_gate.alpha_text.item():.3f} | "
                        f"target='{batch['target_text'][0]}'"
                    )

            avg_jepa = epoch_jepa_sum / max(num_batches, 1)
            avg_lm   = epoch_lm_sum   / max(num_batches, 1)
            avg_var  = vis_var_sum    / max(num_batches, 1)

            print(f"\n{'='*70}")
            print(f"Epoch {epoch_full+1}/{total_epochs}")
            print(f"  jepa_loss : {avg_jepa:.4f}  (↓ = predictor learning)")
            print(f"  lm_loss   : {avg_lm:.4f}    (↓ = language decoder learning)")
            print(f"  vis_var   : {avg_var:.6f}  (>0.001 = no collapse)")
            print(f"  α_vision  : {modality_gate.alpha_vision.item():.3f}")
            print(f"  α_motion  : {modality_gate.alpha_motion.item():.3f}")
            print(f"  α_text    : {modality_gate.alpha_text.item():.3f}")
            print(f"  batches   : {num_batches}")
            print(f"{'='*70}\n")

            checkpoint_name = (
                f"checkpoints/jepa_motion_{USE_MOTION}_{train_split}_{current_file}_{epoch_full}.pt"
            )
            from peft import get_peft_model_state_dict
            lora_sd = get_peft_model_state_dict(llm)

            torch.save({
                "bridge_state_dict":          bridge.state_dict(),
                "context_encoder_state_dict": context_encoder.state_dict(),
                "predictor_state_dict":       predictor.state_dict(),
                "modality_gate_state_dict":   modality_gate.state_dict(),
                "modality_align_state_dict":  modality_align.state_dict(),
                "lora_state_dict":            lora_sd,
                # target_encoder omitted — rebuilt from context_encoder on load
                # optimizer_state_dict omitted — saves ~1-2 GB per checkpoint
                "epoch":                      epoch_full,
            }, checkpoint_name)

        print("Training complete.")

    # =========================================================================
    # INFERENCE
    # =========================================================================
    if INFERENCE_BRIDGE:
        if TRAIN_BRIDGE:
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = (
                f"checkpoints/jepa_motion_True_0.1_main_t-4_lora_47_JEPA_7.pt"
            )

        ckpt = torch.load(checkpoint_path, map_location="cuda")
        bridge.load_state_dict(ckpt["bridge_state_dict"])
        context_encoder.load_state_dict(ckpt["context_encoder_state_dict"])
        predictor.load_state_dict(ckpt["predictor_state_dict"])
        modality_gate.load_state_dict(ckpt["modality_gate_state_dict"])
        if "modality_align_state_dict" in ckpt:
            modality_align.load_state_dict(ckpt["modality_align_state_dict"])
            print("Modality alignment projector loaded.")
        # Re-sync target encoder from loaded context encoder weights.
        for xi, theta in zip(target_encoder.encoder.parameters(),
                             context_encoder.parameters()):
            xi.data.copy_(theta.data)
        print("Target encoder re-synced from context encoder.")

        from peft import set_peft_model_state_dict
        if "lora_state_dict" in ckpt:
            set_peft_model_state_dict(llm, ckpt["lora_state_dict"])
            print("LoRA weights loaded.")

        bridge.eval(); context_encoder.eval()
        predictor.eval(); modality_gate.eval(); modality_align.eval(); llm.eval()

        results = []
        for batch in tqdm(test_loader, desc="Testing"):
            if batch is None:
                continue
            if "history_sig_first" not in batch:
                continue

            hf  = batch["history_sig_first"].to("cuda", dtype=llm.dtype)
            hl  = batch["history_sig_last"].to("cuda",  dtype=llm.dtype)
            he  = batch["history_ego"].to("cuda",       dtype=llm.dtype) if USE_MOTION else None

            predictions = run_inference(hf, hl, he, batch["prompt_text"][0], top_k=1)[0]

            results.append({
                "video_path":        batch["video_info"][0]["video_path"],
                "start_time":        batch["video_info"][0]["start_time"],
                "end_time":          batch["video_info"][0]["end_time"],
                "prompt":            batch["prompt_text"][0],
                "generated_captions": predictions,
                "target_caption":    batch["target_text"][0],
                "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_checkpoint":  checkpoint_path,
                "Motion":            USE_MOTION,
            })

        output_file = f"test_inference_results_jepa_motion_{USE_MOTION}_{current_file}.csv"
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

    if ANALYSE_INFERENCE:
        if INFERENCE_BRIDGE:
            analysis_csv = output_file
        else:
            # Point at a previously saved inference CSV to run analysis standalone
            analysis_csv = f"test_inference_results_jepa_motion_True_main_t-4_lora_47_JEPA.csv"
    
        df_eval           = pd.read_csv(analysis_csv)
        generated_col     = "generated_captions"
        target_col        = "target_caption"
        predictions_eval  = df_eval[generated_col].astype(str).tolist()
        references_eval   = df_eval[target_col].astype(str).tolist()
    
        smooth      = SmoothingFunction().method1
        rouge_sc    = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
        bleu1_scores  = []
        bleu2_scores  = []
        bleu3_scores  = []
        bleu4_scores  = []
        meteor_scores = []
        rouge_scores  = []
    
        for pred, ref in tqdm(zip(predictions_eval, references_eval),
                            total=len(predictions_eval), desc="Scoring"):
            pred_tok = nltk.word_tokenize(pred.lower())
            ref_tok  = nltk.word_tokenize(ref.lower())
    
            bleu1_scores.append(sentence_bleu([ref_tok], pred_tok, weights=(1,0,0,0),
                                            smoothing_function=smooth))
            bleu2_scores.append(sentence_bleu([ref_tok], pred_tok, weights=(0.5,0.5,0,0),
                                            smoothing_function=smooth))
            bleu3_scores.append(sentence_bleu([ref_tok], pred_tok, weights=(0.33,0.33,0.33,0),
                                            smoothing_function=smooth))
            bleu4_scores.append(sentence_bleu([ref_tok], pred_tok, weights=(0.25,0.25,0.25,0.25),
                                            smoothing_function=smooth))
            meteor_scores.append(meteor_score([ref_tok], pred_tok))
            rouge_scores.append(rouge_sc.score(ref, pred)["rougeL"].fmeasure)
    
        print("\nEvaluation Results:")
        print(f"BLEU-1 : {sum(bleu1_scores)/len(bleu1_scores):.4f}")
        print(f"BLEU-2 : {sum(bleu2_scores)/len(bleu2_scores):.4f}")
        print(f"BLEU-3 : {sum(bleu3_scores)/len(bleu3_scores):.4f}")
        print(f"BLEU-4 : {sum(bleu4_scores)/len(bleu4_scores):.4f}")
        print(f"METEOR : {sum(meteor_scores)/len(meteor_scores):.4f}")
        print(f"ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.4f}")
    
        df_eval["BLEU-1"]  = bleu1_scores
        df_eval["BLEU-2"]  = bleu2_scores
        df_eval["BLEU-3"]  = bleu3_scores
        df_eval["BLEU-4"]  = bleu4_scores
        df_eval["METEOR"]  = meteor_scores
        df_eval["ROUGE-L"] = rouge_scores
        df_eval.to_csv(f"analysis_{analysis_csv}", index=False)
        print(f"Saved scored CSV → analysis_{analysis_csv}")
    
        # =========================================================================
        # VISUALIZE  —  6 figures adapted for the JEPA architecture
        #
        # Key difference from lora_46/47:
        #   OLD: bridge → temporal_resampler → LLM
        #   NEW: bridge → context_encoder (s_x) → predictor (s_hat) → LLM
        #
        # All figures therefore route through context_encoder + predictor.
        # Fig 6 is new: visualises the JEPA representation alignment
        #   (cosine-sim between s_x, s_hat, and s_y per sample).
        # =========================================================================
    
    if VISUALIZE_FRAMES:
        # ------------------------------------------------------------------
        # Load checkpoint  (skip if we just finished training)
        # ------------------------------------------------------------------
        if TRAIN_BRIDGE:
            checkpoint_path = f"checkpoints/jepa_motion_{USE_MOTION}_{train_split}_{current_file}_{epoch_full}.pt"
        else:
            checkpoint_path = f"checkpoints/jepa_motion_True_0.1_main_t-4_lora_47_JEPA_7.pt"

        ckpt_vis = torch.load(
            checkpoint_path,
            map_location="cuda"
        )
        bridge.load_state_dict(ckpt_vis["bridge_state_dict"])
        context_encoder.load_state_dict(ckpt_vis["context_encoder_state_dict"])
        predictor.load_state_dict(ckpt_vis["predictor_state_dict"])
        modality_gate.load_state_dict(ckpt_vis["modality_gate_state_dict"])
        if "modality_align_state_dict" in ckpt_vis:
            modality_align.load_state_dict(ckpt_vis["modality_align_state_dict"])
        from peft import set_peft_model_state_dict
        if "lora_state_dict" in ckpt_vis:
            set_peft_model_state_dict(llm, ckpt_vis["lora_state_dict"])
        # Re-sync target encoder from loaded context encoder
        for xi, theta in zip(target_encoder.encoder.parameters(),
                            context_encoder.parameters()):
            xi.data.copy_(theta.data)

        bridge.eval(); context_encoder.eval()
        predictor.eval(); modality_gate.eval(); modality_align.eval()
        target_encoder.encoder.eval(); llm.eval()

        vis_save_dir = f"modality_visualizations/jepa_{analysis_csv}"
        os.makedirs(vis_save_dir, exist_ok=True)

        # ==================================================================
        # RUN ALL FIGURES  (n_samples from test_loader)
        # ==================================================================
        N_VIS_SAMPLES = 5
        samples_vis   = []
        for batch_v in test_loader:
            if batch_v is None:
                continue
            if "history_sig_first" not in batch_v:
                continue
            samples_vis.append(batch_v)
            if len(samples_vis) >= N_VIS_SAMPLES:
                break

        print(f"\nGenerating JEPA visualizations for {len(samples_vis)} samples → {vis_save_dir}/\n")

        for i, batch_v in enumerate(samples_vis):
            prefix = os.path.join(vis_save_dir, f"sample_{i+1:02d}")
            tgt    = batch_v["target_text"][0]
            print(f"[Sample {i+1}/{len(samples_vis)}] target='{tgt}'")

            for fn, label in [
                (fig1_jepa_attention_split,          "fig1_attention_split"),
                (fig2_jepa_temporal_heatmap,         "fig2_temporal_heatmap"),
                (fig3_jepa_ablation,                 "fig3_ablation"),
                (fig4_jepa_pca,                      "fig4_pca"),
                (fig5_jepa_text_saliency,            "fig5_text_saliency"),
                (fig6_jepa_representation_alignment, "fig6_jepa_alignment"),
            ]:
                try:
                    fn(batch_v, save_path=f"{prefix}_{label}.png")
                except Exception as e:
                    print(f"  {label} failed: {e}")
            print()

        print(f"All figures saved → {vis_save_dir}/")

        # ==================================================================
        # COLLAPSE DIAGNOSTIC  —  token variance + PCA health check
        # ==================================================================
        print("\n── Collapse diagnostic ──")
        cache_dir_diag = "extracted_features_new"
        all_pt = [f for f in os.listdir(cache_dir_diag) if f.endswith(".pt")]
        if all_pt:
            raw = torch.load(os.path.join(cache_dir_diag, all_pt[1]), map_location="cpu")
            sig = raw.get("sig_last_features", raw.get("sig_features",
                        list(raw.values())[0]))
            if sig.dim() == 2:
                sig = sig.unsqueeze(0)   # [1, 729, 1152]
            sig = sig.to("cuda", dtype=llm.dtype)
            with torch.no_grad():
                out = bridge.siglip_proj.float()(sig.float()).to(llm.dtype)
            tokens = out[0].float().cpu().numpy()   # [729, D]

            token_var = np.var(tokens, axis=0).mean()
            print(f"  Mean token variance : {token_var:.8f}")
            print(f"  (>0.001 = healthy | <0.000001 = collapsed)")

            pca_diag = PCA(n_components=5)
            pca_diag.fit(tokens)
            print(f"  PCA explained variance (first 5 PCs):")
            for j, r in enumerate(pca_diag.explained_variance_ratio_):
                print(f"    PC{j+1}: {r*100:.2f}%")
            print("  (PC1 ~100% = collapsed | PC1 <80% = healthy)")

            # Also check s_x token variance across the full test set (first 50 batches)
            sx_vars = []
            n_checked = 0
            for batch_d in test_loader:
                if batch_d is None or "history_sig_first" not in batch_d:
                    continue
                with torch.no_grad():
                    all_steps_d, _, _, txt_d = _jepa_bridge_all_steps(batch_d)
                    s_x_d = context_encoder(all_steps_d, txt_d, modality_gate)
                sx_vars.append(s_x_d[0].float().var(dim=0).mean().item())
                n_checked += 1
                if n_checked >= 50:
                    break
            mean_sx_var = np.mean(sx_vars)
            print(f"\n  s_x token variance (avg over {n_checked} batches): {mean_sx_var:.6f}")
            print(f"  (>0.001 = healthy | <0.0001 = representation collapse)")
        else:
            print(f"  No .pt files found in '{cache_dir_diag}' — skipping.")
