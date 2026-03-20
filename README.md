# Bordelius_Taetz_MM_WHN
MultiModel "What happens next" framework
<img width="731" height="1029" alt="image" src="https://github.com/user-attachments/assets/a9def789-c2b9-4bbf-a158-7a9ae3b3ee3d" />

### Architecture Details

| Component | Model / Class | Input | Output | Frozen? | Key Detail |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SigLIP** | `siglip-so400m-patch14-384` | 1 frame 384x384 per timestep | 729 tokens x 1152-D | Yes | Pure patch tokens, no CLS. 1 middle frame per clip. |
| **EgoVLP** | SpaceTimeTransformer `base_patch16_224` | 4 frames 224x224 per timestep | 785 tokens x 768-D | Yes | Joint space-time attention. CLS at index 0 = global motion. |
| **MultimodalBridge (vision)** | 4-layer MLP `siglip_proj` | 729 x 1152-D | 729 x 3584-D | No (lr 1e-4) | float32 projection fix. Prevents LayerNorm saturation under bfloat16. |
| **MultimodalBridge (motion)** | 4-layer MLP `egovlp_proj` | 8 x 768-D (slice `[:8]`) | 8 x 3584-D | No (lr 2e-5) | Same float32 fix. `+motion_type_embed` tags tokens as motion. |
| **Concat per timestep** | `torch.cat` | 729 + 8 tokens | 737 x 3584-D per step | - | Vision and motion fused before resampler. |
| **TemporalPerceiverResampler** | Custom Perceiver IO (depth=2, heads=8) | 4 steps x 737 tokens = 2948 tokens | 256 x 3584-D | No (lr 1e-4) | Self-attn + cross-attn latents. `time_embed` stamps each step. |
| **Qwen 2.5 7B-Instruct** | `AutoModelForCausalLM` (4-bit NF4) | 256 visual + text tokens | Predicted action tokens | Yes (base) | Loaded float16 compute dtype. `use_cache=False` during training. |
| **LoRA adapters** | r=16, alpha=32, dropout=0.05 | Applied to all 7 attn+FFN layers | ~0.5% trainable params | No (lr 2e-5) | Teaches Qwen to interpret visual tokens without overwriting language knowledge. |



Motion = True:
Trainable Parameters in Bridge: 32,603,648
Trainable Parameters in Resampler: 412,116,992
Trainable Parameters in LLM (LoRA): 40,370,176
TOTAL Trainable Parameters: 485,090,816

Motion = False:
Trainable Parameters in Bridge: 16,988,160
Trainable Parameters in Resampler: 412,116,992
Trainable Parameters in LLM (LoRA): 40,370,176
TOTAL Trainable Parameters: 469,475,328
