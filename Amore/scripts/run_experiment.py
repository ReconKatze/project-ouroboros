#!/usr/bin/env python3
"""Step 4: d_state gradient experiment — 4-variant head-to-head comparison.

Trains four Mamba hybrid configurations under identical distillation conditions
and compares validation loss to answer: does an exponential d_state schedule
(Variant D) outperform uniform d_state (Variant A)?

Key improvements over v1:
  - Data pre-tokenized once, shared across all variants (fair + fast)
  - Held-out validation set evaluated every --eval-every steps (clean signal)
  - EMA-smoothed training loss display
  - LR warmup (first 200 steps) to stabilise variant D's larger layers
  - Optional gradient accumulation (--grad-accum) for even smoother updates
  - Comparison table driven by val loss, not noisy per-step training loss

Variants:
    A          — uniform small:   d_state=16  for all 18 Mamba layers
    B          — uniform large:   d_state=64  for all 18 Mamba layers
    C          — three-tier:      d_state 16 (×6), 64 (×6), 128 (×6)
    D_constant — exponential [16→256] + per-layer constant gate sigmoid(γ_i)
    D_proper   — exponential [16→256] + shared-β depth gate sigmoid(β·depth + γ_i)

Usage (Colab A100):
    python scripts/run_experiment.py --steps 10000 --batch-size 8
    python scripts/run_experiment.py --steps 10000 --batch-size 8 --variants D_constant,D_proper

AMP: bfloat16 on A100/H100 (no GradScaler needed), float16 on T4 (GradScaler).
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from chimera.models.hybrid_model import HybridChimeraModel, MambaGuidedAttentionWrapper
from chimera.models.beta_mamba import BetaGatedMamba, BetaGated2AMamba, PSoftMambaWrapper
from chimera.utils.layer_plan import build_layer_plan, ATTN_KEEP_DEFAULTS


# ---------------------------------------------------------------------------
# AMP dtype detection
# ---------------------------------------------------------------------------

def get_amp_dtype():
    """bfloat16 on A100/H100 (native HW, no scaler needed), float16 on T4/V100."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def _snap16(x: float) -> int:
    """Round x to the nearest multiple of 16 (minimum 16).

    Mamba-3 requires headdim_qk (= d_state) to be even; Triton kernels prefer
    multiples of 16. Snapping keeps all our target values (32, 64, 96, 128)
    unchanged while fixing bell/exponential intermediate values (e.g. 53 → 48).
    """
    return max(16, round(x / 16) * 16)


def make_d_states(spec: dict, n_mamba: int, spans: list | None = None) -> list:
    """Generate a d_state list for n_mamba layers from a schedule spec.

    Specs:
        {"type": "uniform",             "d": 64}
        {"type": "exponential",         "d_min": 64, "d_max": 256}
        {"type": "three_tier"}           -- uses d=32/64/128 split evenly across thirds
        {"type": "bell",                "d_min": 64, "d_max": 256}
            Single U-shape across all Mamba layers.
        {"type": "bell_per_span",       "d_min": 64, "d_max": 256}
            Independent U-shape per attention span. Requires spans=.
        {"type": "bell_per_span_ramped","d_min": 64, "d_max": 256, "tail": [224, 256]}
            Per-span bell with fixed high-capacity tail layers (preparation phase).
            Last len(tail) layers of each span are set to tail values. Requires spans=.
    """
    t = spec["type"]
    if t == "uniform":
        return [spec["d"]] * n_mamba
    elif t == "exponential":
        d_min, d_max = spec["d_min"], spec["d_max"]
        return [
            _snap16(d_min * (d_max / d_min) ** (i / max(n_mamba - 1, 1)))
            for i in range(n_mamba)
        ]
    elif t == "bell":
        # Single U-shape across all Mamba layers: d_max at edges, d_min at centre.
        d_min, d_max = spec["d_min"], spec["d_max"]
        return [
            _snap16(d_max - (d_max - d_min) *
                    (1 - math.cos(2 * math.pi * i / max(n_mamba - 1, 1))) / 2)
            for i in range(n_mamba)
        ]
    elif t == "bell_per_span":
        # Independent U-shape within each attention span.
        # d_max at both ends of every span: absorption after attention + preparation before it.
        if spans is None:
            raise ValueError("bell_per_span requires spans= (list of Mamba counts per span)")
        d_min, d_max = spec["d_min"], spec["d_max"]
        result = []
        for span_len in spans:
            for i in range(span_len):
                d = _snap16(d_max - (d_max - d_min) *
                            (1 - math.cos(2 * math.pi * i / max(span_len - 1, 1))) / 2)
                result.append(d)
        return result
    elif t == "bell_per_span_ramped":
        # Per-span bell with explicit high-capacity tail for the preparation phase.
        # The last len(tail) layers of each span are pinned to tail values (e.g. [96, 128]).
        # Remaining layers follow a standard bell curve snapped to multiples of 16.
        # Rationale: H reads relevance from the pre-attention Mamba state — giving those
        # layers maximum capacity improves H's token selection signal directly.
        if spans is None:
            raise ValueError("bell_per_span_ramped requires spans= argument")
        d_min, d_max = spec["d_min"], spec["d_max"]
        tail = spec.get("tail", [])
        result = []
        for span_len in spans:
            body_len = max(span_len - len(tail), 1)
            for i in range(body_len):
                d = _snap16(d_max - (d_max - d_min) *
                            (1 - math.cos(2 * math.pi * i / max(body_len - 1, 1))) / 2)
                result.append(d)
            result.extend(tail[:span_len - body_len])
        return result
    elif t == "three_tier":
        tier = n_mamba // 3
        return [32] * tier + [64] * tier + [128] * (n_mamba - 2 * tier)
    else:
        raise ValueError(f"Unknown d_state_spec type: {t!r}")


VARIANTS = {
    "A": {
        "d_state_spec": {"type": "uniform", "d": 16},
        "gate_mode":    None,
        "label":        "Uniform small (d=16)",
    },
    "B": {
        "d_state_spec": {"type": "uniform", "d": 32},
        "gate_mode":    None,
        "label":        "Uniform (d=32)",
    },
    "C": {
        "d_state_spec": {"type": "three_tier"},
        "gate_mode":    None,
        "label":        "Three-tier (d=16/64/128)",
    },
    "D_constant": {
        "d_state_spec": {"type": "exponential", "d_min": 32, "d_max": 128},
        "gate_mode":    "constant",
        "label":        "Exp gradient (32→128) + constant gate",
    },
    "D_proper": {
        "d_state_spec": {"type": "exponential", "d_min": 32, "d_max": 128},
        "gate_mode":    "shared_beta",
        "label":        "Exp gradient (32→128) + shared-β depth gate",
    },
    "D_bell": {
        "d_state_spec": {"type": "bell", "d_min": 32, "d_max": 128},
        "gate_mode":    "constant",
        "label":        "Global bell (128→32→128) + constant gate",
    },
    "D_bell2": {
        "d_state_spec": {"type": "bell_per_span", "d_min": 32, "d_max": 64},
        "gate_mode":    "constant",
        "label":        "Per-span bell (64→32→64 each span) + constant gate",
    },
    "D_bell2_32": {
        "d_state_spec": {"type": "bell_per_span_ramped", "d_min": 32, "d_max": 64,
                         "tail": [96, 128]},
        "gate_mode":    "constant",
        "label":        "Per-span bell asymmetric (64→32→64→96→128 each span) + constant gate",
    },
    "D_exp_inv": {
        "d_state_spec": {"type": "exponential", "d_min": 128, "d_max": 32},
        "gate_mode":    "constant",
        "label":        "Inv. exponential (128→32) + constant gate",
    },
    "B_gated": {
        "d_state_spec": {"type": "uniform", "d": 64},
        "gate_mode":    "constant",
        "label":        "Uniform (d=64) + constant gate",
    },
    "B_psoft": {
        "d_state_spec": {"type": "uniform", "d": 64},
        "gate_mode":    "constant",
        "psoft":        True,
        "label":        "Uniform (d=64) + constant gate + P_soft",
    },
    "B_psoft_2A": {
        "d_state_spec": {"type": "uniform", "d": 64},
        "gate_mode":    "constant",
        "gate_2a":      True,
        "psoft":        True,
        "label":        "B_psoft + 2A temporal/block gates",
    },
    "B_psoft_2H": {
        "d_state_spec": {"type": "uniform", "d": 64},
        "gate_mode":    "constant",
        "psoft":        True,
        "gate_h":       True,
        "label":        "B_psoft + 2H Mamba-guided sparse attention",
    },
    "D_bell2_32_psoft": {
        # Floor raised to 64: body=[64×6] + tail=[96,128] per span.
        # Only the 2 pre-attention layers per span get elevated d_state.
        "d_state_spec": {"type": "bell_per_span_ramped", "d_min": 64, "d_max": 64,
                         "tail": [96, 128]},
        "gate_mode":    "constant",
        "psoft":        True,
        "label":        "Per-span d=64 + tail [96,128] + P_soft",
    },
}


# ---------------------------------------------------------------------------
# Data: pre-tokenise once, share across all variants
# ---------------------------------------------------------------------------

def load_data(tokenizer, seq_len, n_train, n_val):
    """Pre-tokenise wikitext-103 train and test splits into fixed-length chunks.

    Returns:
        train_chunks: list of n_train LongTensors, shape [seq_len]
        val_chunks:   list of n_val   LongTensors, shape [seq_len]
    """
    def _collect(split, n):
        ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1",
            split=split, streaming=True, trust_remote_code=False,
        )
        chunks, buf = [], []
        for ex in ds:
            if len(chunks) >= n:
                break
            text = ex["text"].strip()
            if not text:
                continue
            buf.extend(tokenizer.encode(text, add_special_tokens=False))
            while len(buf) >= seq_len and len(chunks) < n:
                chunks.append(torch.tensor(buf[:seq_len], dtype=torch.long))
                buf = buf[seq_len:]
        return chunks

    print(f"Pre-tokenising data (train={n_train}, val={n_val}, seq_len={seq_len})...")
    train_chunks = _collect("train", n_train)
    val_chunks   = _collect("test",  n_val)
    print(f"  Loaded {len(train_chunks)} train chunks, {len(val_chunks)} val chunks.")
    return train_chunks, val_chunks


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def distill_loss(student_logits, teacher_logits, T=2.0):
    """Temperature-scaled KL with per-token std normalisation.

    Handles teacher/student vocab size mismatch (e.g. Qwen2.5-7B has 152064
    tokens vs 1.5B's 151936) by truncating to the smaller vocab.
    """
    vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    student_logits = student_logits[..., :vocab]
    teacher_logits = teacher_logits[..., :vocab]
    s = student_logits / (student_logits.std(dim=-1, keepdim=True) + 1e-6)
    t = teacher_logits / (teacher_logits.std(dim=-1, keepdim=True) + 1e-6)
    log_p_s = F.log_softmax(s / T, dim=-1)
    p_t     = F.softmax(t / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, name, step, student, optimizer, scheduler,
                    val_log, ema_loss, chunk_idx):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "variant":         name,
        "step":            step,
        "model_state":     student.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_log":         val_log,
        "ema_loss":        ema_loss,
        "chunk_idx":       chunk_idx,
    }, path)
    print(f"  [{name}] Checkpoint saved → {path}")


def load_checkpoint(path, name, student, optimizer, scheduler, args):
    """Load checkpoint. Returns (start_step, val_log, ema_loss, chunk_idx).

    If the saved step equals args.steps the checkpoint is a *finished* run —
    only model weights are restored (warm-start for Round 2). Otherwise full
    state is restored so an interrupted run continues from where it left off.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved_step = ckpt.get("step", 0)
    is_crash_recovery = saved_step > 0 and saved_step < args.steps
    # Crash recovery: strict=True (all keys must match exactly).
    # Warm-start: strict=False so new parameters (e.g. 2A gates) keep fresh init.
    student.load_state_dict(ckpt["model_state"], strict=is_crash_recovery)
    if is_crash_recovery:
        # Crash-recovery: restore full training state
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        print(f"  [{name}] Resumed from step {saved_step}/{args.steps} (crash recovery)")
        return (saved_step,
                ckpt.get("val_log", {}),
                ckpt.get("ema_loss", None),
                ckpt.get("chunk_idx", 0))
    else:
        # Warm-start: model weights only, reset optimizer/scheduler/counters
        print(f"  [{name}] Warm-started from '{path}' (model weights only)")
        return 0, {}, None, 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(student, teacher, val_chunks, device, temperature, amp_dtype):
    """Compute mean distillation loss on the held-out val set (no grad)."""
    student.eval()
    total = 0.0
    with torch.no_grad():
        for chunk in val_chunks:
            input_ids = chunk.unsqueeze(0).to(device)
            teacher_logits = teacher(input_ids=input_ids).logits
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                full_logits    = student(input_ids)
                student_logits = full_logits[:, student.num_sinks:, :]
            loss = distill_loss(student_logits.float(), teacher_logits.float(), temperature)
            total += loss.item()
    student.train()
    return total / len(val_chunks)


# ---------------------------------------------------------------------------
# Student builder
# ---------------------------------------------------------------------------

def build_variant_student(args, cfg, device):
    """Build HybridChimeraModel for one variant with the correct d_state schedule."""
    import torch.nn as nn

    base = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.float32
    ).to(device)

    # Resolve attention layer indices — explicit override or model-type default
    if args.attn_indices is not None:
        attn_keep = {int(x) for x in args.attn_indices.split(",")}
    else:
        attn_keep = None  # build_layer_plan uses model-type preset

    effective_attn = (
        attn_keep if attn_keep is not None
        else ATTN_KEEP_DEFAULTS.get(base.config.model_type, {0, 3, 7, 11})
    )
    n_mamba  = base.config.num_hidden_layers - len(effective_attn)

    # bell_per_span needs span sizes derived from the attention layer positions
    spans = None
    if cfg["d_state_spec"]["type"] in ("bell_per_span", "bell_per_span_ramped"):
        n_layers   = base.config.num_hidden_layers
        mamba_set  = set(range(n_layers)) - effective_attn
        boundaries = [-1] + sorted(effective_attn) + [n_layers]
        spans = [
            sum(1 for l in mamba_set if boundaries[i] < l < boundaries[i + 1])
            for i in range(len(boundaries) - 1)
        ]
        spans = [s for s in spans if s > 0]  # drop empty gaps (e.g. adjacent attn layers)

    d_states = make_d_states(cfg["d_state_spec"], n_mamba, spans=spans)
    print(f"  Attn layers: {sorted(effective_attn)}  "
          f"({len(effective_attn)} attn, {n_mamba} Mamba) | "
          f"d_state: {d_states[0]}→{d_states[-1]}")

    layer_plan = build_layer_plan(
        base.config.num_hidden_layers,
        attn_keep_indices=attn_keep,
        model_type=base.config.model_type,
        d_state=d_states,
    )

    hybrid = HybridChimeraModel(base, layer_plan, num_sinks=4, device=str(device))

    # Gate variants: wrap Mamba blocks with BetaGatedMamba (or 2A) BEFORE freeze
    gate_mode = cfg.get("gate_mode")
    gate_2a   = cfg.get("gate_2a", False)
    if gate_mode in ("constant", "shared_beta"):
        total_mamba = sum(1 for p in hybrid.layer_plan if p["kind"] == "mamba")
        shared_beta = nn.Parameter(torch.zeros(1)) if gate_mode == "shared_beta" else None
        d_model = base.config.hidden_size

        if gate_2a:
            hybrid.gate2a_sparsity_accum = []
            hybrid.gate2a_stats_accum    = []

        mamba_idx = 0
        for plan in hybrid.layer_plan:
            if plan["kind"] == "mamba":
                wrapper = hybrid.adapter.get_attention(hybrid.layers[plan["layer_idx"]])
                if gate_2a:
                    wrapper.mamba = BetaGated2AMamba(
                        wrapper.mamba, mamba_idx, total_mamba,
                        d_model=d_model,
                        d_state=d_states[mamba_idx],
                        sparsity_accum=hybrid.gate2a_sparsity_accum,
                        stats_accum=hybrid.gate2a_stats_accum,
                        shared_beta=shared_beta,
                    )
                else:
                    wrapper.mamba = BetaGatedMamba(
                        wrapper.mamba, mamba_idx, total_mamba, shared_beta=shared_beta
                    )
                mamba_idx += 1
        if shared_beta is not None:
            hybrid.register_parameter("shared_beta", shared_beta)

    # P_soft: wrap BetaGatedMamba with error-driven input (must be AFTER gate wrapping)
    # Creates a shared loss accumulator on the model; training loop reads it each step.
    hybrid.psoft_loss_accum = []
    if cfg.get("psoft"):
        d_model = base.config.hidden_size
        for plan in hybrid.layer_plan:
            if plan["kind"] == "mamba":
                wrapper = hybrid.adapter.get_attention(hybrid.layers[plan["layer_idx"]])
                wrapper.mamba = PSoftMambaWrapper(
                    wrapper.mamba, d_model, loss_accum=hybrid.psoft_loss_accum
                )

    # Round 2H: Mamba-guided sparse attention
    # Wraps attention layers (except layer 0) with MambaGuidedAttentionWrapper.
    # Preceding Mamba reference is wired so each guided layer reads _last_out
    # from the Mamba wrapper immediately before it in the layer sequence.
    gate_h = cfg.get("gate_h", False)
    if gate_h:
        hybrid.h_align_accum = []
        hybrid.h_stats_accum = []
        hybrid.h_step_ref    = [0]   # updated by training loop before each forward

        last_mamba = None
        for plan in hybrid.layer_plan:
            layer = hybrid.layers[plan["layer_idx"]]
            if plan["kind"] == "mamba":
                last_mamba = hybrid.adapter.get_attention(layer)
            elif plan["kind"] == "attention" and last_mamba is not None:
                orig_attn = hybrid.adapter.get_attention(layer)
                guided = MambaGuidedAttentionWrapper(
                    orig_attn,
                    d_model=d_model,
                    align_accum=hybrid.h_align_accum,
                    stats_accum=hybrid.h_stats_accum,
                    step_ref=hybrid.h_step_ref,
                )
                guided.preceding_mamba = last_mamba
                hybrid.adapter.set_attention(layer, guided)

    # Freeze all, then unfreeze Mamba wrappers + sink tokens
    for p in hybrid.parameters():
        p.requires_grad_(False)
    for plan in hybrid.layer_plan:
        if plan["kind"] == "mamba":
            wrapper = hybrid.adapter.get_attention(hybrid.layers[plan["layer_idx"]])
            for p in wrapper.parameters():
                p.requires_grad_(True)
    hybrid.sink_tokens.sinks.requires_grad_(True)
    # shared_beta is registered on hybrid (not on any wrapper) — unfreeze explicitly
    if hasattr(hybrid, "shared_beta"):
        hybrid.shared_beta.requires_grad_(True)
    # H: unfreeze only the relevance projections (original_attn stays frozen)
    if gate_h:
        for plan in hybrid.layer_plan:
            if plan["kind"] == "attention":
                wrapper = hybrid.adapter.get_attention(hybrid.layers[plan["layer_idx"]])
                if isinstance(wrapper, MambaGuidedAttentionWrapper):
                    wrapper.W_q_rel.weight.requires_grad_(True)
                    wrapper.W_k_rel.weight.requires_grad_(True)

    hybrid = hybrid.to(device)
    hybrid.train()

    trainable = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in hybrid.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")
    return hybrid


# ---------------------------------------------------------------------------
# Training loop for one variant
# ---------------------------------------------------------------------------

def _alpha_summary(student, cfg) -> str:
    """Return a compact one-line string of α values grouped by Mamba span.

    Example:  α  [.485 .467 .450 .446 .447 .451 .480 .461] | [.444 .460 ... .467] | [.467 ... .485]
    The last value in each bracket is the layer just before an attention boundary.
    """
    gate_mode = cfg.get("gate_mode")
    if gate_mode not in ("constant", "shared_beta"):
        return ""

    # Collect (layer_idx, alpha) for all Mamba layers in order
    entries = []
    for plan in student.layer_plan:
        if plan["kind"] == "mamba":
            wrapper = student.adapter.get_attention(student.layers[plan["layer_idx"]])
            bg = wrapper.mamba
            if gate_mode == "shared_beta":
                a = torch.sigmoid(student.shared_beta * bg.depth + bg.gamma).item()
            else:
                a = torch.sigmoid(bg.gamma).item()
            entries.append((plan["layer_idx"], a))

    if not entries:
        return ""

    # Group into spans separated by attention boundaries
    # A new span starts whenever layer indices are non-consecutive
    spans, current = [], []
    for i, (li, a) in enumerate(entries):
        if i > 0 and li != entries[i - 1][0] + 1:
            spans.append(current)
            current = []
        current.append(a)
    if current:
        spans.append(current)

    span_strs = ["[" + " ".join(f"{a:.3f}" for a in s) + "]" for s in spans]
    return "α  " + " | ".join(span_strs)


def train_variant(name, cfg, args, teacher, train_chunks, val_chunks, device, amp_dtype, effective_lr):
    """Train one variant. Returns dict of val losses keyed by step."""
    print(f"\n{'='*60}")
    print(f"Variant {name}: {cfg['label']}")
    print(f"{'='*60}")

    student          = build_variant_student(args, cfg, device)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer        = AdamW(trainable_params, lr=effective_lr, weight_decay=0.01)
    use_scaler       = (amp_dtype == torch.float16)
    scaler           = torch.amp.GradScaler("cuda") if use_scaler else None

    # Warmup then cosine decay: ramps to peak LR over warmup_steps, then decays to ~0
    warmup_steps = min(200, args.steps // 10)
    total_steps  = args.steps

    def lr_schedule(s):
        if warmup_steps > 0 and s < warmup_steps:
            return (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    val_log     = {}
    ema_loss    = None
    ema_alpha   = 0.95
    chunk_idx   = 0
    accum_steps = 0
    start_step  = 0

    # Resume from checkpoint if requested
    if args.resume:
        start_step, val_log, ema_loss, chunk_idx = load_checkpoint(
            args.resume, name, student, optimizer, scheduler, args
        )

    # Checkpoint output path (save_dir/<name>_final.pt and mid-run)
    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Initial validation (skip if resuming mid-run — already have val_log)
    if start_step == 0:
        print(f"  [{name}] Step 0 — initial val loss...")
        val_loss = evaluate(student, teacher, val_chunks, device, args.temperature, amp_dtype)
        val_log[0] = val_loss
        print(f"  [{name}] Step    0/{args.steps} | val={val_loss:.4f}  (pre-training)")

    for step in range(start_step + 1, args.steps + 1):
        batch_chunks = [
            train_chunks[(chunk_idx + i) % len(train_chunks)]
            for i in range(args.batch_size)
        ]
        input_ids = torch.stack(batch_chunks).to(device)   # [batch_size, seq_len]
        chunk_idx += args.batch_size

        # Update H step counter before forward so MambaGuidedAttentionWrapper
        # sees the correct phase (dense vs sparse)
        if hasattr(student, "h_step_ref"):
            student.h_step_ref[0] = step

        # Teacher forward
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids).logits

        # Student forward
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            full_logits    = student(input_ids)
            student_logits = full_logits[:, student.num_sinks:, :]

        loss = distill_loss(
            student_logits.float(),
            teacher_logits.float(),
            T=args.temperature,
        )

        # P_soft prediction loss — accumulated by PSoftMambaWrapper during forward
        psoft_accum = getattr(student, "psoft_loss_accum", [])
        if psoft_accum:
            loss_pc = sum(psoft_accum) / len(psoft_accum)
            loss = loss + args.lambda_pc * loss_pc
            psoft_accum.clear()

        # Round 2A gate sparsity loss — accumulated by BetaGated2AMamba during forward
        gate2a_sparsity = getattr(student, "gate2a_sparsity_accum", [])
        if gate2a_sparsity:
            loss = loss + sum(gate2a_sparsity)
            gate2a_sparsity.clear()

        # Round 2H alignment loss — entropy reg on relevance distributions
        h_align = getattr(student, "h_align_accum", [])
        if h_align:
            loss = loss + sum(h_align) / len(h_align)
            h_align.clear()

        if not torch.isfinite(loss):
            print(f"  [{name}] Step {step}: NaN/Inf loss — stopping variant.")
            break

        # Accumulate gradients (scale by grad_accum so effective LR is consistent)
        if use_scaler:
            scaler.scale(loss / args.grad_accum).backward()
        else:
            (loss / args.grad_accum).backward()
        accum_steps += 1

        if accum_steps == args.grad_accum:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum_steps = 0

        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else ema_alpha * ema_loss + (1 - ema_alpha) * loss_val

        # Mid-run checkpoint
        if save_dir and args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            ckpt_path = os.path.join(save_dir, f"{name}_step{step}.pt")
            save_checkpoint(ckpt_path, name, step, student, optimizer, scheduler,
                            val_log, ema_loss, chunk_idx)

        # Validation eval
        if step % args.eval_every == 0:
            val_loss = evaluate(student, teacher, val_chunks, device, args.temperature, amp_dtype)
            val_log[step] = val_loss
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [{name}] Step {step:5d}/{args.steps} | "
                  f"train_ema={ema_loss:8.3f} | val={val_loss:8.3f} | lr={lr_now:.2e}")
            summary = _alpha_summary(student, cfg)
            if summary:
                print(f"    [{name}] {summary}")
            gate2a_stats = getattr(student, "gate2a_stats_accum", [])
            if gate2a_stats:
                mean_gt  = sum(s["g_time_mean"]    for s in gate2a_stats) / len(gate2a_stats)
                mean_sil = sum(s["g_time_silence"] for s in gate2a_stats) / len(gate2a_stats)
                print(f"    [{name}] g_time: mean={mean_gt:.3f}  silent={mean_sil*100:.1f}%")
                gate2a_stats.clear()
            h_stats = getattr(student, "h_stats_accum", [])
            if h_stats:
                mean_ent = sum(s["entropy"] for s in h_stats) / len(h_stats)
                phase    = h_stats[-1]["phase"]
                label    = "dense" if phase == 0 else "sparse"
                print(f"    [{name}] H: entropy={mean_ent:.3f}  ({label})")
                h_stats.clear()

        elif step % args.log_every == 0 or step == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [{name}] Step {step:5d}/{args.steps} | "
                  f"train_ema={ema_loss:8.3f} | lr={lr_now:.2e}")

    # Final checkpoint
    if save_dir:
        final_path = os.path.join(save_dir, f"{name}_final.pt")
        save_checkpoint(final_path, name, args.steps, student, optimizer, scheduler,
                        val_log, ema_loss, chunk_idx)

    # Final validation if not already done at last step
    if args.steps not in val_log:
        val_loss = evaluate(student, teacher, val_chunks, device, args.temperature, amp_dtype)
        val_log[args.steps] = val_loss
        print(f"  [{name}] Step {args.steps:5d}/{args.steps} | val={val_loss:.4f}  (final)")

    # Gate variants: report learned β, γ, α per Mamba layer
    gate_mode = cfg.get("gate_mode")
    if gate_mode in ("constant", "shared_beta"):
        if gate_mode == "shared_beta":
            shared_b = student.shared_beta.item()
            print(f"\n  Shared β = {shared_b:+.4f}")
        print(f"  Learned γ and α per Mamba layer:")
        for plan in student.layer_plan:
            if plan["kind"] == "mamba":
                wrapper = student.adapter.get_attention(student.layers[plan["layer_idx"]])
                bg = wrapper.mamba
                g  = bg.gamma.item()
                if gate_mode == "shared_beta":
                    alpha = torch.sigmoid(student.shared_beta * bg.depth + bg.gamma).item()
                    print(f"    layer {plan['layer_idx']:2d} depth={bg.depth:.2f}: "
                          f"γ={g:+.3f}  →  α={alpha:.3f}")
                else:
                    alpha = torch.sigmoid(bg.gamma).item()
                    print(f"    layer {plan['layer_idx']:2d}: γ={g:+.3f}  →  α={alpha:.3f}")

    del student, optimizer, scheduler
    if scaler is not None:
        del scaler
    torch.cuda.empty_cache()

    return val_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Step 4: d_state gradient experiment")
    p.add_argument("--teacher",     default="Qwen/Qwen2.5-7B")
    p.add_argument("--student",     default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--steps",       type=int,   default=5000)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--seq-len",     type=int,   default=1024)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--grad-accum",  type=int,   default=1,
                   help="Gradient accumulation steps (effective batch = grad_accum × 1)")
    p.add_argument("--eval-every",  type=int,   default=250,
                   help="Evaluate on val set every N steps (clean signal)")
    p.add_argument("--log-every",   type=int,   default=100,
                   help="Log training EMA loss every N steps")
    p.add_argument("--n-val",       type=int,   default=100,
                   help="Number of held-out val chunks")
    p.add_argument("--batch-size",  type=int,   default=4,
                   help="Sequences per gradient step. Default 4 suits A100 80GB comfortably.")
    p.add_argument("--variants",     default="D_constant,D_proper",
                   help="Comma-separated variants to run, e.g. 'B,D_constant' or 'A,B,C,D_constant,D_proper'")
    p.add_argument("--attn-indices", default=None,
                   help="Comma-separated attention layer indices, e.g. '0,11,23' for 3-attention "
                        "variant. Default: model-type preset ({0,9,18,27} for Qwen2.5-1.5B).")
    p.add_argument("--save-dir",        default="checkpoints",
                   help="Directory to save checkpoints. Final: <name>_final.pt. "
                        "Set to '' to disable saving.")
    p.add_argument("--resume",          default=None,
                   help="Path to checkpoint to load before training. If step < --steps, "
                        "resumes full state (crash recovery). If step >= --steps, loads "
                        "model weights only (warm-start for Round 2).")
    p.add_argument("--checkpoint-every", type=int, default=2500,
                   help="Save mid-run checkpoint every N steps (0 = final only).")
    p.add_argument("--lambda-pc",        type=float, default=0.01,
                   help="P_soft prediction loss weight. 0.01 validated in B_psoft Run 2.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    variant_names = [v.strip() for v in args.variants.split(",")]
    selected = {k: VARIANTS[k] for k in variant_names if k in VARIANTS}
    unknown = [k for k in variant_names if k not in VARIANTS]
    if unknown:
        print(f"Unknown variants: {unknown}. Available: {list(VARIANTS.keys())}")
    if not selected:
        print("No valid variants selected.")
        return

    # --- AMP dtype ---
    amp_dtype  = get_amp_dtype()
    dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
    print(f"AMP dtype: {dtype_name}  |  GradScaler: {'no (bfloat16)' if amp_dtype == torch.bfloat16 else 'yes'}")

    n_train = args.steps * args.batch_size + 500   # enough chunks, never repeat within a variant

    # Linear LR scaling: base --lr was tuned at batch_size=1
    effective_lr = args.lr * args.batch_size
    print(f"LR: {args.lr} × batch_size {args.batch_size} → effective LR {effective_lr:.2e}")
    print(f"\nVariants: {list(selected.keys())}  |  Steps each: {args.steps}  |  "
          f"Grad accum: {args.grad_accum}  |  Val every: {args.eval_every}")

    # Load teacher once
    print(f"\nLoading teacher: {args.teacher} ({dtype_name})...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=amp_dtype
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("Teacher loaded.")

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-tokenise once, shared across all variants
    train_chunks, val_chunks = load_data(tokenizer, args.seq_len, n_train, args.n_val)

    # Run all selected variants
    all_val_logs = {}
    for name, cfg in selected.items():
        all_val_logs[name] = train_variant(
            name, cfg, args, teacher, train_chunks, val_chunks, device, amp_dtype, effective_lr
        )

    # Comparison table (val loss)
    print("\n" + "=" * 72)
    print("EXPERIMENT RESULTS  (validation loss — lower is better)")
    print("=" * 72)

    # Collect final val losses
    results = {}
    for name, val_log in all_val_logs.items():
        sorted_steps = sorted(val_log.keys())
        mid_step  = sorted_steps[len(sorted_steps) // 2]
        results[name] = {
            "label":      VARIANTS[name]["label"],
            "val_start":  val_log.get(0,           float("nan")),
            "val_mid":    val_log.get(mid_step,     float("nan")),
            "val_final":  val_log[sorted_steps[-1]],
        }

    baseline = results.get("A", {}).get("val_final", None)

    print(f"{'Var':<4} {'d_state schedule':<32} {'Val@0':>8} {'Val@mid':>8} {'Val@end':>8} {'vs A':>8}")
    print("-" * 72)
    for name, r in results.items():
        if baseline and math.isfinite(r["val_final"]) and math.isfinite(baseline) and name != "A":
            pct   = (r["val_final"] - baseline) / baseline * 100
            vs_a  = f"{pct:+.1f}%"
        elif name == "A":
            vs_a = "(base)"
        else:
            vs_a = "N/A"
        label_s = r["label"][:30]
        print(f"{name:<4} {label_s:<32} {r['val_start']:>8.2f} "
              f"{r['val_mid']:>8.2f} {r['val_final']:>8.2f} {vs_a:>8}")

    print("=" * 72)

    # Go/no-go: B vs D_constant answers whether exponential d_state beats uniform
    if "D_constant" in results and "B" in results:
        b_val  = results["B"]["val_final"]
        dc_val = results["D_constant"]["val_final"]
        if math.isfinite(b_val) and math.isfinite(dc_val):
            improvement = (b_val - dc_val) / b_val * 100
            print(f"\nD_constant vs B: {improvement:+.2f}%")
            if improvement >= 10.0:
                print("GO — Exponential d_state clearly outperforms uniform. Use for 9B Chimera.")
            elif improvement > 2.0:
                print("WEAK SIGNAL — D_constant better but under 10% threshold.")
            elif improvement > 0:
                print("MARGINAL — Within noise floor. d_state schedule has no clear effect.")
            else:
                print("NULL RESULT — d_state schedule has no effect at this configuration.")

    # Go/no-go: D_proper vs D_constant answers whether depth-aware gating adds value
    if "D_proper" in results and "D_constant" in results:
        dc = results["D_constant"]["val_final"]
        dp = results["D_proper"]["val_final"]
        if math.isfinite(dc) and math.isfinite(dp):
            improvement = (dc - dp) / dc * 100
            print(f"\nD_proper improvement over D_constant: {improvement:+.2f}%")
            if improvement > 0:
                print("RESULT: Shared-β depth gate adds value — use depth-aware gating for 9B Chimera.")
            else:
                print("RESULT: Depth-awareness adds no value over per-layer constants.")
                print("        Use sigmoid(γ_i) per layer; drop β entirely.")
    elif "A" in results:
        best = min(results, key=lambda k: results[k]["val_final"])
        a = results["A"]["val_final"]
        b = results[best]["val_final"]
        if math.isfinite(a) and math.isfinite(b):
            improvement = (a - b) / a * 100
            print(f"\nBest variant: {best} — {improvement:.1f}% improvement over A")
            if improvement >= 10.0:
                print("GO — Use this d_state schedule for 9B Chimera.")
            else:
                print("NOTE — No variant reached ≥10% improvement threshold over A.")

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
