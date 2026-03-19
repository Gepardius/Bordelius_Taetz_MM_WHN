"""
visualize_modalities.py
-----------------------
Drop-in visualization suite for the egocentric action-prediction model.
Produces 5 figures that qualitatively show how EgoVLP (motion), SigLIP (vision)
and the text history each influence the model's understanding.

Usage (paste after your model/checkpoint is loaded in main):
    from visualize_modalities import run_modality_visualizations
    run_modality_visualizations(test_loader, n_samples=5)

Prerequisites (already in your environment):
    matplotlib, seaborn, numpy, torch, sklearn (for PCA/TSNE)

All figures are saved as PNG files and also shown interactively if a display
is available. Pass save_dir="your/path" to control the output folder.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _make_save_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _collect_samples(dataloader, n_samples):
    """Pull the first n_samples non-None batches from a DataLoader."""
    samples = []
    for batch in dataloader:
        if batch is None:
            continue
        samples.append(batch)
        if len(samples) >= n_samples:
            break
    return samples


def _bridge_forward_with_hooks(bridge, sig_t, ego_t, use_motion):
    """
    Run one timestep through the bridge and return:
        vis_tokens   : [1, 729, D]  — spatial SigLIP tokens in LLM space
        mot_tokens   : [1, 8,  D]  — motion EgoVLP tokens in LLM space (or None)
        fused_tokens : [1, 737, D] — concatenated output (or [1,729,D])

    Uses float32 for the projection MLPs to prevent the bfloat16 rank-1 collapse
    that caused PCA PC2=0.0% — same fix as in MultimodalBridge.forward().
    The module weights are temporarily cast to float32 for computation only;
    the result is cast back to the original input dtype before returning.
    """
    orig_dtype = sig_t.dtype
    vis_tokens = bridge.siglip_proj.float()(sig_t.float()).to(orig_dtype)  # [1, 729, D]

    if use_motion and ego_t is not None and bridge.use_motion:
        ego_sliced = ego_t[:, :8, :]                                              # [1, 8, 768]
        mot_tokens = bridge.egovlp_proj.float()(ego_sliced.float()).to(orig_dtype)# [1, 8, D]
        mot_tokens = mot_tokens + bridge.motion_type_embed
        fused = torch.cat([vis_tokens, mot_tokens], dim=1)
        return vis_tokens, mot_tokens, fused
    else:
        return vis_tokens, None, vis_tokens


def _resampler_forward_with_attn(temporal_resampler, all_visual_steps, llm_dtype):
    """
    Re-run the resampler and capture cross-attention weights from both layers.
    Returns:
        latents   : [1, 256, D]
        attn_maps : list of [1, num_heads, 256, total_tokens]   (one per layer)
    """
    B = all_visual_steps[0].shape[0]
    processed = []
    for t_idx, step in enumerate(all_visual_steps):
        processed.append(step + temporal_resampler.time_embed[t_idx])
    x = torch.cat(processed, dim=1)                           # [1, T*tokens, D]

    latents = temporal_resampler.latents.unsqueeze(0).expand(B, -1, -1).clone()
    attn_maps = []

    for layer in temporal_resampler.layers:
        # Self-attention
        sa_out, _ = layer["self_attn"](latents, latents, latents)
        latents = layer["norm0"](latents + sa_out)

        # Cross-attention — need_weights=True to get the attention matrix
        cross_in = layer["norm1"](latents)
        attn_out, attn_w = layer["cross_attn"](
            cross_in, x, x,
            need_weights=True,
            average_attn_weights=False       # keep per-head: [B, H, Q, K]
        )
        attn_maps.append(attn_w.detach().cpu().float())      # [1, H, 256, T*tokens]
        latents = latents + attn_out
        latents = latents + layer["ff"](latents)

    return temporal_resampler.norm(latents), attn_maps


def _get_prompt_token_labels(prompt_text, tokenizer, max_tokens=30):
    """Return readable token strings for the prompt, truncated for display."""
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
    labels = [tokenizer.decode([i]).replace("Ġ", " ").strip() or f"[{i}]" for i in ids]
    # Keep only the last max_tokens (the history part is most informative)
    if len(labels) > max_tokens:
        labels = ["..."] + labels[-max_tokens:]
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Cross-attention heatmap: which tokens the resampler attends to
# ──────────────────────────────────────────────────────────────────────────────

def fig1_resampler_attention_heatmap(batch, bridge, temporal_resampler, llm,
                                     tokenizer, use_motion, save_path):
    """
    Shows: for each of the 4 timesteps, how much total attention (summed over
    all 256 latents and all heads) falls on (a) SigLIP spatial tokens vs
    (b) EgoVLP motion tokens.
    Reveals whether motion tokens are actually being attended to.
    """
    bridge.eval(); temporal_resampler.eval()

    hist_sig = batch["history_sig"].to("cuda", dtype=llm.dtype)
    hist_ego = batch["history_ego"].to("cuda", dtype=llm.dtype) if use_motion else None
    prompt_text = batch["prompt_text"][0]
    target = batch["target_text"][0]

    with torch.no_grad():
        all_steps = []
        step_labels = []          # tag each token with its source
        for t in range(4):
            sig_t = hist_sig[:, t]
            ego_t = hist_ego[:, t] if hist_ego is not None else None
            vis, mot, fused = _bridge_forward_with_hooks(bridge, sig_t, ego_t, use_motion)
            all_steps.append(fused)
            n_vis = vis.shape[1]   # 729
            n_mot = mot.shape[1] if mot is not None else 0
            step_labels += [f"t-{4-t}_SIG"] * n_vis + [f"t-{4-t}_EGO"] * n_mot

        _, attn_maps = _resampler_forward_with_attn(temporal_resampler, all_steps, llm.dtype)

    # Average both layers, all heads, all latent queries → one weight per key token
    # attn_maps: list of [1, H, 256, K]
    combined = torch.stack([a[0].mean(dim=[0, 1]) for a in attn_maps], dim=0).mean(0)  # [K]
    combined = combined / combined.sum()

    n_tokens = len(step_labels)
    assert len(combined) == n_tokens, f"Mismatch {len(combined)} vs {n_tokens}"

    # Aggregate by timestep × modality
    timesteps = [f"t-{4-t}" for t in range(4)]
    sig_weights = np.zeros(4)
    ego_weights = np.zeros(4)
    ptr = 0
    for t in range(4):
        sig_t = hist_sig[:, t]
        ego_t = hist_ego[:, t] if hist_ego is not None else None
        n_vis = 729
        n_mot = 8 if (use_motion and ego_t is not None) else 0
        sig_weights[t] = combined[ptr:ptr + n_vis].sum().item()
        ptr += n_vis
        if n_mot:
            ego_weights[t] = combined[ptr:ptr + n_mot].sum().item()
            ptr += n_mot

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Fig 1 — Resampler attention by modality\nPrompt: …{prompt_text[-80:]}\nTarget: {target}",
        fontsize=9, ha="left", x=0.02
    )

    # Bar chart: per-timestep modality split
    x = np.arange(4)
    w = 0.35
    axes[0].bar(x - w/2, sig_weights * 100, w, label="SigLIP (vision)", color="#4C8EDA")
    axes[0].bar(x + w/2, ego_weights * 100, w, label="EgoVLP (motion)", color="#E07B54")
    axes[0].set_xticks(x); axes[0].set_xticklabels(timesteps)
    axes[0].set_ylabel("% of total attention"); axes[0].set_ylim(0, 50)
    axes[0].set_title("Attention share per timestep")
    axes[0].legend(fontsize=8)

    # Pie: overall SIG vs EGO
    total_sig = sig_weights.sum()
    total_ego = ego_weights.sum()
    labels_pie = ["SigLIP vision", "EgoVLP motion"] if use_motion else ["SigLIP vision"]
    sizes_pie = [total_sig, total_ego] if use_motion else [total_sig]
    colors_pie = ["#4C8EDA", "#E07B54"][:len(sizes_pie)]
    axes[1].pie(sizes_pie, labels=labels_pie, colors=colors_pie, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 9})
    axes[1].set_title("Overall vision vs motion attention split")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Temporal attention heatmap (which timesteps matter most)
# ──────────────────────────────────────────────────────────────────────────────

def fig2_temporal_attention_heatmap(batch, bridge, temporal_resampler, llm,
                                    tokenizer, use_motion, save_path):
    """
    Shows: a 256×4 heatmap where each row is one latent query and each column
    is a timestep. Brighter = the latent attended strongly to that time window.
    Reveals whether the model focuses on recent vs distant history.
    """
    bridge.eval(); temporal_resampler.eval()

    hist_sig = batch["history_sig"].to("cuda", dtype=llm.dtype)
    hist_ego = batch["history_ego"].to("cuda", dtype=llm.dtype) if use_motion else None
    target = batch["target_text"][0]

    with torch.no_grad():
        all_steps = []
        tokens_per_step = []
        for t in range(4):
            sig_t = hist_sig[:, t]
            ego_t = hist_ego[:, t] if hist_ego is not None else None
            _, _, fused = _bridge_forward_with_hooks(bridge, sig_t, ego_t, use_motion)
            all_steps.append(fused)
            tokens_per_step.append(fused.shape[1])   # 729 or 737

        _, attn_maps = _resampler_forward_with_attn(temporal_resampler, all_steps, llm.dtype)

    # Average over layers and heads → [256, total_tokens]
    avg_attn = torch.stack([a[0].mean(dim=[0]) for a in attn_maps], dim=0).mean(0)  # [256, K]

    # Split K into 4 timestep buckets and sum within each bucket → [256, 4]
    per_timestep = np.zeros((avg_attn.shape[0], 4))
    ptr = 0
    for t, n in enumerate(tokens_per_step):
        per_timestep[:, t] = avg_attn[:, ptr:ptr + n].sum(-1).float().numpy()
        ptr += n
    # Normalise each latent row to [0,1]
    row_max = per_timestep.max(axis=1, keepdims=True).clip(min=1e-9)
    per_timestep = per_timestep / row_max

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(per_timestep, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Timestep"); ax.set_ylabel("Latent index (0–255)")
    ax.set_xticks([0,1,2,3]); ax.set_xticklabels(["t-4","t-3","t-2","t-1"])
    ax.set_title(f"Fig 2 — Per-latent temporal attention\nTarget: {target}", fontsize=9)
    plt.colorbar(im, ax=ax, label="Relative attention (row-normalised)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Modality ablation: confidence change when each modality is zeroed
# ──────────────────────────────────────────────────────────────────────────────

def fig3_modality_ablation_bar(batch, bridge, temporal_resampler, llm,
                               tokenizer, use_motion, save_path):
    """
    Shows: the log-probability of the ground-truth target token sequence under
    three conditions: (1) all modalities, (2) vision zeroed, (3) motion zeroed.
    The drop in confidence quantifies each modality's contribution.
    """
    bridge.eval(); temporal_resampler.eval()

    hist_sig = batch["history_sig"].to("cuda", dtype=llm.dtype)
    hist_ego = batch["history_ego"].to("cuda", dtype=llm.dtype) if use_motion else None
    prompt_text = batch["prompt_text"][0]
    target_text = batch["target_text"][0]

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids.to("cuda")

    def compute_logprob(sig_override=None, ego_override=None):
        with torch.no_grad():
            all_steps = []
            for t in range(4):
                sig_t = hist_sig[:, t] if sig_override is None else sig_override
                ego_t = (hist_ego[:, t] if hist_ego is not None else None) \
                        if ego_override is None else ego_override
                _, _, fused = _bridge_forward_with_hooks(bridge, sig_t, ego_t, use_motion)
                all_steps.append(fused)
            vis_emb = temporal_resampler(all_steps)
            p_emb = llm.base_model.model.model.embed_tokens(prompt_ids)
            t_emb = llm.base_model.model.model.embed_tokens(target_ids)
            full = torch.cat([vis_emb, p_emb, t_emb], dim=1)
            ignore = vis_emb.shape[1] + p_emb.shape[1]
            labels = torch.full((1, full.shape[1]), -100, dtype=torch.long, device="cuda")
            labels[0, ignore:] = target_ids[0]
            out = llm(inputs_embeds=full, labels=labels)
            return -out.loss.item()   # higher = more confident

    # Full model
    lp_full = compute_logprob()

    # Zero SigLIP: replace spatial tokens with zeros
    zero_sig = torch.zeros_like(hist_sig[:, 0])
    lp_no_vision = compute_logprob(sig_override=zero_sig)

    # Zero EgoVLP: replace motion input with zeros
    if use_motion and hist_ego is not None:
        zero_ego = torch.zeros_like(hist_ego[:, 0])
        lp_no_motion = compute_logprob(ego_override=zero_ego)
    else:
        lp_no_motion = lp_full   # motion not active

    # Zero text history: rebuild prompt with masked history
    masked_prompt = prompt_text.replace(
        "Previous actions:", "Previous actions: None available."
    ).split("t-4:")[0] + f"Predict the next action.:<|im_end|>\n<|im_start|>assistant\n"
    masked_ids = tokenizer(masked_prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        all_steps = []
        for t in range(4):
            sig_t = hist_sig[:, t]
            ego_t = hist_ego[:, t] if hist_ego is not None else None
            _, _, fused = _bridge_forward_with_hooks(bridge, sig_t, ego_t, use_motion)
            all_steps.append(fused)
        vis_emb = temporal_resampler(all_steps)
        p_emb = llm.base_model.model.model.embed_tokens(masked_ids)
        t_emb = llm.base_model.model.model.embed_tokens(target_ids)
        full = torch.cat([vis_emb, p_emb, t_emb], dim=1)
        ignore = vis_emb.shape[1] + p_emb.shape[1]
        labels = torch.full((1, full.shape[1]), -100, dtype=torch.long, device="cuda")
        labels[0, ignore:] = target_ids[0]
        lp_no_text = -llm(inputs_embeds=full, labels=labels).loss.item()

    conditions = ["Full model", "Vision zeroed\n(SigLIP=0)", "Motion zeroed\n(EgoVLP=0)", "Text history\nmasked"]
    logprobs   = [lp_full, lp_no_vision, lp_no_motion, lp_no_text]
    drops      = [lp_full - lp for lp in logprobs]
    colors = ["#5BAD72", "#4C8EDA", "#E07B54", "#9B59B6"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Fig 3 — Modality ablation (confidence on target)\nTarget: '{target_text}'",
        fontsize=9, ha="left", x=0.02
    )
    # Absolute log-probs
    axes[0].bar(conditions, logprobs, color=colors, alpha=0.85)
    axes[0].axhline(lp_full, color="black", linestyle="--", linewidth=0.8, label="Full model")
    axes[0].set_ylabel("Log-probability (higher = more confident)")
    axes[0].set_title("Absolute confidence per condition")
    axes[0].legend(fontsize=8)
    for i, v in enumerate(logprobs):
        axes[0].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    # Drop from full model
    axes[1].bar(conditions, drops, color=colors, alpha=0.85)
    axes[1].set_ylabel("Confidence drop (full − ablated)")
    axes[1].set_title("Importance = drop when modality is removed")
    for i, v in enumerate(drops):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — PCA of visual latents colored by modality source
# ──────────────────────────────────────────────────────────────────────────────

def fig4_pca_token_space(batch, bridge, llm, use_motion, save_path):
    """
    Shows: a 2-D PCA scatter of all tokens entering the resampler for one
    sample. SigLIP tokens (×4 timesteps) and EgoVLP tokens are coloured
    differently, with timestep encoded by marker shade.
    Reveals whether motion tokens live in a distinct region of token space.
    """
    bridge.eval()

    hist_sig = batch["history_sig"].to("cuda", dtype=llm.dtype)
    hist_ego = batch["history_ego"].to("cuda", dtype=llm.dtype) if use_motion else None
    target = batch["target_text"][0]

    all_tokens = []
    labels_mod = []    # "SIG" or "EGO"
    labels_time = []   # 0..3

    with torch.no_grad():
        for t in range(4):
            sig_t = hist_sig[:, t]
            ego_t = hist_ego[:, t] if hist_ego is not None else None
            vis, mot, _ = _bridge_forward_with_hooks(bridge, sig_t, ego_t, use_motion)
            all_tokens.append(vis[0].float().cpu().numpy())   # [729, D]
            labels_mod  += ["SIG"] * 729
            labels_time += [t] * 729
            if mot is not None:
                all_tokens.append(mot[0].float().cpu().numpy())  # [8, D]
                labels_mod  += ["EGO"] * 8
                labels_time += [t] * 8

    X = np.concatenate(all_tokens, axis=0)   # [N, D]
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)

    labels_mod  = np.array(labels_mod)
    labels_time = np.array(labels_time)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Fig 4 — PCA of resampler input tokens\nTarget: {target}", fontsize=9)

    # Left: coloured by modality
    sig_mask = labels_mod == "SIG"
    ego_mask = labels_mod == "EGO"
    axes[0].scatter(X2[sig_mask, 0], X2[sig_mask, 1], s=1, alpha=0.3,
                    color="#4C8EDA", label=f"SigLIP ({sig_mask.sum()})")
    if ego_mask.any():
        axes[0].scatter(X2[ego_mask, 0], X2[ego_mask, 1], s=20, alpha=0.9,
                        color="#E07B54", marker="*", zorder=5,
                        label=f"EgoVLP ({ego_mask.sum()})")
    axes[0].set_title("Token space by modality")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].legend(markerscale=4, fontsize=8)

    # Right: coloured by timestep
    cmap = plt.cm.plasma
    for t in range(4):
        mask = labels_time == t
        axes[1].scatter(X2[mask & sig_mask, 0], X2[mask & sig_mask, 1],
                        s=1, alpha=0.25, color=cmap(t / 3), label=f"t-{4-t}")
        if ego_mask.any():
            axes[1].scatter(X2[mask & ego_mask, 0], X2[mask & ego_mask, 1],
                            s=30, alpha=1.0, color=cmap(t / 3), marker="*", zorder=5)
    axes[1].set_title("Token space by timestep (★ = EgoVLP)")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].legend(markerscale=4, fontsize=8, title="Timestep")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Text token attention from the LLM's perspective
# ──────────────────────────────────────────────────────────────────────────────

def fig5_text_token_importance(batch, bridge, temporal_resampler, llm,
                               tokenizer, use_motion, save_path):
    """
    Shows: input-gradient saliency of each text token in the prompt w.r.t.
    the target log-probability. Higher saliency = that token mattered more.
    Reveals which parts of the history (t-4 vs t-1, verb vs noun) drive predictions.
    """
    bridge.eval(); temporal_resampler.eval()

    hist_sig = batch["history_sig"].to("cuda", dtype=llm.dtype)
    hist_ego = batch["history_ego"].to("cuda", dtype=llm.dtype) if use_motion else None
    prompt_text = batch["prompt_text"][0]
    target_text = batch["target_text"][0]

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids.to("cuda")

    # Decode token labels
    raw_labels = [tokenizer.decode([i]).replace("Ġ", " ") for i in prompt_ids[0]]
    # Keep last 40 tokens for readability
    MAX_SHOW = 40
    if len(raw_labels) > MAX_SHOW:
        raw_labels = ["..."] + raw_labels[-MAX_SHOW:]
        prompt_ids_show = prompt_ids[:, -MAX_SHOW:]
    else:
        prompt_ids_show = prompt_ids

    with torch.no_grad():
        all_steps = []
        for t in range(4):
            sig_t = hist_sig[:, t]
            ego_t = hist_ego[:, t] if hist_ego is not None else None
            _, _, fused = _bridge_forward_with_hooks(bridge, sig_t, ego_t, use_motion)
            all_steps.append(fused)
        vis_emb = temporal_resampler(all_steps)

    # Enable gradients for the prompt embeddings only
    p_emb = llm.base_model.model.model.embed_tokens(prompt_ids).detach().requires_grad_(True)
    t_emb = llm.base_model.model.model.embed_tokens(target_ids).detach()

    full = torch.cat([vis_emb.detach(), p_emb, t_emb], dim=1)
    ignore = vis_emb.shape[1] + p_emb.shape[1]
    labels = torch.full((1, full.shape[1]), -100, dtype=torch.long, device="cuda")
    labels[0, ignore:] = target_ids[0]

    out = llm(inputs_embeds=full, labels=labels)
    (-out.loss).backward()

    # Gradient × embedding magnitude as saliency → [seq_len]
    grad = p_emb.grad                          # [1, seq, D]
    saliency = (grad * p_emb).abs().sum(-1)[0].float().cpu().numpy()  # [seq]

    # Trim to displayed tokens
    if len(saliency) > MAX_SHOW:
        saliency = saliency[-MAX_SHOW:]
    saliency = saliency / saliency.max().clip(min=1e-9)

    fig, ax = plt.subplots(figsize=(max(10, len(raw_labels) * 0.4), 3.5))
    bars = ax.bar(range(len(raw_labels)), saliency,
                  color=plt.cm.Reds(saliency), edgecolor="none")
    ax.set_xticks(range(len(raw_labels)))
    ax.set_xticklabels(raw_labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Gradient saliency (normalised)")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"Fig 5 — Text token importance (gradient saliency)\n"
        f"Target: '{target_text}'  |  Prompt tail shown",
        fontsize=9
    )
    sm = ScalarMappable(cmap="Reds", norm=Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Saliency", shrink=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def run_modality_visualizations(
    dataloader,
    bridge,
    temporal_resampler,
    llm,
    tokenizer,
    use_motion,
    n_samples=5,
    save_dir="modality_visualizations"
):
    """
    Run all 5 visualizations on n_samples batches from dataloader.

    Parameters
    ----------
    dataloader        : torch DataLoader (test or val split)
    bridge            : MultimodalBridge instance (loaded + eval)
    temporal_resampler: TemporalPerceiverResampler instance (loaded + eval)
    llm               : PEFT-wrapped Qwen LLM (loaded + eval)
    tokenizer         : AutoTokenizer for Qwen
    use_motion        : bool — matches USE_MOTION flag used at inference
    n_samples         : how many samples to visualize (default 5)
    save_dir          : folder where PNGs are saved
    """
    _make_save_dir(save_dir)
    samples = _collect_samples(dataloader, n_samples)
    print(f"\nGenerating modality visualizations for {len(samples)} samples → {save_dir}/\n")

    for i, batch in enumerate(samples):
        prefix = os.path.join(save_dir, f"sample_{i+1:02d}")
        target = batch["target_text"][0] if isinstance(batch["target_text"], list) else batch["target_text"][0]
        print(f"[Sample {i+1}/{len(samples)}] target='{target}'")

        try:
            fig1_resampler_attention_heatmap(
                batch, bridge, temporal_resampler, llm, tokenizer, use_motion,
                save_path=f"{prefix}_fig1_attention_split.png"
            )
        except Exception as e:
            print(f"  Fig1 failed: {e}")

        try:
            fig2_temporal_attention_heatmap(
                batch, bridge, temporal_resampler, llm, tokenizer, use_motion,
                save_path=f"{prefix}_fig2_temporal_heatmap.png"
            )
        except Exception as e:
            print(f"  Fig2 failed: {e}")

        try:
            fig3_modality_ablation_bar(
                batch, bridge, temporal_resampler, llm, tokenizer, use_motion,
                save_path=f"{prefix}_fig3_ablation.png"
            )
        except Exception as e:
            print(f"  Fig3 failed: {e}")

        try:
            fig4_pca_token_space(
                batch, bridge, llm, use_motion,
                save_path=f"{prefix}_fig4_pca.png"
            )
        except Exception as e:
            print(f"  Fig4 failed: {e}")

        try:
            fig5_text_token_importance(
                batch, bridge, temporal_resampler, llm, tokenizer, use_motion,
                save_path=f"{prefix}_fig5_text_saliency.png"
            )
        except Exception as e:
            print(f"  Fig5 failed: {e}")

        print()

    print(f"Done. All figures saved in: {save_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# HOW TO CALL THIS FROM YOUR MAIN SCRIPT
# ──────────────────────────────────────────────────────────────────────────────
# Paste these lines into main_t-4_lora_36.py (or _37.py) after the checkpoint
# is loaded and bridge/temporal_resampler/llm are in eval() mode:
#
#   from visualize_modalities import run_modality_visualizations
#   run_modality_visualizations(
#       dataloader         = test_loader,
#       bridge             = bridge,
#       temporal_resampler = temporal_resampler,
#       llm                = llm,
#       tokenizer          = tokenizer,
#       use_motion         = USE_MOTION,
#       n_samples          = 5,
#       save_dir           = "modality_visualizations"
#   )