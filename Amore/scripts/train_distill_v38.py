#!/usr/bin/env python3
"""Single-variant V3.8 distillation with full telemetry capture.

V3.8 changes vs V3:
  - Architecture: d_model=5120, 32 layers (28 Mamba + 4 attention), anchors (0,10,21,31), ~6.5B params
  - Teacher: Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled (27B dense; Claude Opus
    4.6 reasoning distilled into Qwen3.5-27B). Loaded in 4-bit NF4 (bitsandbytes) via
    --teacher-4bit (default True). At bf16 the teacher consumes ~54 GB; 4-bit brings it to
    ~13.5 GB, leaving comfortable headroom alongside the 6.5B student.
  - Optimizer: 8-bit AdamW (bitsandbytes) via --use-8bit-adam (strongly recommended).
    Saves ~42 GB of optimizer state vs float32 Adam (6.5B × 8 bytes → 6.5B × 2 bytes).
    VRAM breakdown: teacher ~13.5 GB + student weights ~13 GB + grads ~13 GB + 8-bit Adam ~13 GB
    + activations ~5 GB ≈ 58 GB total.
  - Training data: --dataset-mix wiki (default) or diverse (WikiText + C4 + GitHub code +
    OpenWebMath). Use 'diverse' to better match the teacher's code/reasoning expertise.

This runner is for diagnosis-heavy distillation. It prints a compact summary for
each logged step and writes full per-step telemetry bundles containing inputs,
teacher logits, student logits, losses, diagnostics, state tensors, and gradient
norms.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import os
import sys
from pathlib import Path
from dataclasses import replace
import json
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "V3.8"))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_BNB_IMPORT_ERROR: Exception | None = None
_BNB_IMPORT_LOG = ""
try:
    _bnb_stdout = io.StringIO()
    _bnb_stderr = io.StringIO()
    with redirect_stdout(_bnb_stdout), redirect_stderr(_bnb_stderr):
        import bitsandbytes as bnb
    _BNB_IMPORT_LOG = (_bnb_stdout.getvalue() + _bnb_stderr.getvalue()).strip()
    _BNB_AVAILABLE = True
except Exception as exc:
    _BNB_IMPORT_ERROR = exc
    _BNB_IMPORT_LOG = (_bnb_stdout.getvalue() + _bnb_stderr.getvalue()).strip()
    _BNB_AVAILABLE = False

from life_eq_v38.factory import build_config, build_model
from life_eq_v38.forensics import (
    ForensicConfig,
    ForensicEventManager,
    build_full_forensic_snapshot,
    build_lightweight_replay_entry,
)
from life_eq_v38.model import ForwardOutputs
from life_eq_v38.state import FullState
from life_eq_v38.telemetry import (
    render_snapshot_summary,
)
from chimera.evaluation.runner import EvalRunner


def get_amp_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError("train_distill_v35.py is GPU-only. CUDA is required.")
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def bitsandbytes_requirement_error(feature_name: str) -> RuntimeError:
    """Build a concrete, actionable bitsandbytes failure message."""
    lines = [
        f"{feature_name} requires bitsandbytes, but bitsandbytes could not be loaded.",
        f"torch.__version__={torch.__version__}",
        f"torch.version.cuda={torch.version.cuda}",
    ]
    if torch.version.cuda and str(torch.version.cuda).startswith("13"):
        lines.append(
            "torch is on a CUDA 13 build. In Colab this usually means a package install "
            "upgraded torch away from the runtime-supported CUDA stack."
        )
        lines.append(
            "Rebuild the environment from a fresh runtime and ensure mamba-ssm is installed "
            "with --no-deps so it does not upgrade torch."
        )
    if _BNB_IMPORT_ERROR is not None:
        lines.append(f"bitsandbytes import error: {_BNB_IMPORT_ERROR}")
        if "libnvJitLink.so.13" in str(_BNB_IMPORT_ERROR):
            lines.append(
                "This usually means the current PyTorch/bitsandbytes stack is targeting CUDA 13, "
                "but the runtime does not provide the CUDA 13 nvJitLink library."
            )
            lines.append(
                "Rebuild the Colab environment so torch and bitsandbytes match the runtime CUDA "
                "toolchain before rerunning V3.8 training."
            )
    lines.append(
        "Verify the environment first with: "
        "python -c \"import torch; print(torch.__version__, torch.version.cuda)\""
    )
    lines.append(
        "Then verify bitsandbytes with: "
        "python -c \"import bitsandbytes as bnb; print(bnb.__version__)\""
    )
    if _BNB_IMPORT_LOG:
        lines.append("bitsandbytes import log:")
        lines.append(_BNB_IMPORT_LOG)
    return RuntimeError("\n".join(lines))


# ── Dataset mix definitions ──────────────────────────────────────────────────
# Each entry: (hf_path, config_name, hf_split, sampling_weight, text_field)
# text_field is the key in each example dict that holds the raw text content.
# Validation always uses WikiText-103 test split regardless of training mix.
DATASET_MIXES: dict[str, list[tuple]] = {
    "wiki": [
        ("wikitext", "wikitext-103-raw-v1", "train", 1.0, "text"),
    ],
    "diverse": [
        ("wikitext",                     "wikitext-103-raw-v1", "train", 0.15, "text"),
        ("allenai/c4",                   "en",                   "train", 0.35, "text"),
        ("HuggingFaceFW/fineweb",         "CC-MAIN-2024-10",      "train", 0.35, "text"),
        ("open-web-math/open-web-math",  None,                   "train", 0.15, "text"),
    ],
}


def make_token_chunks(tokenizer, seq_len: int, dataset_mix: str = "wiki"):
    from datasets import interleave_datasets as _interleave
    specs = DATASET_MIXES[dataset_mix]

    if len(specs) == 1:
        path, name, split, _, text_field = specs[0]
        ds = load_dataset(path, name, split=split, streaming=True, trust_remote_code=False)
        ds = ds.map(lambda ex, tf=text_field: {"text": ex.get(tf, "") or ""})
    else:
        sub_datasets, weights = [], []
        for path, name, split, weight, text_field in specs:
            d = load_dataset(path, name, split=split, streaming=True, trust_remote_code=False)
            d = d.map(lambda ex, tf=text_field: {"text": ex.get(tf, "") or ""})
            sub_datasets.append(d)
            weights.append(weight)
        total = sum(weights)
        probs = [w / total for w in weights]
        ds = _interleave(sub_datasets, probabilities=probs, seed=42,
                         stopping_strategy="all_exhausted")

    buf = []
    for ex in ds:
        text = ex["text"].strip()
        if not text:
            continue
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(buf) >= seq_len:
            yield torch.tensor(buf[:seq_len], dtype=torch.long)
            buf = buf[seq_len:]


def kl_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    s = student_logits[..., :vocab]
    t = teacher_logits[..., :vocab]
    s = s / (s.std(dim=-1, keepdim=True) + 1e-6)
    t = t / (t.std(dim=-1, keepdim=True) + 1e-6)
    log_p_s = F.log_softmax(s / T, dim=-1)
    p_t = F.softmax(t / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)


def detach_state(state: FullState) -> FullState:
    result = {}
    for k, v in vars(state).items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach()
        elif k == "Z_ssm" and v is not None:
            # Z_ssm is a list of Optional[Tensor]; each tensor is already detached
            # when stored in Mamba3Block._last_final_states, so pass through unchanged.
            result[k] = v
        else:
            result[k] = v
    return FullState(**result)


def _serialize_le_state(state: FullState) -> dict:
    result = {}
    for k, v in vars(state).items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu()
        elif k == "Z_ssm" and v is not None:
            result[k] = [t.detach().cpu() if t is not None else None for t in v]
        else:
            result[k] = v
    return result


def _deserialize_le_state(saved: dict, device: torch.device) -> FullState:
    result = {}
    for k, v in saved.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif k == "Z_ssm" and v is not None:
            result[k] = [t.to(device) if t is not None else None for t in v]
        else:
            result[k] = v
    return FullState(**result)


_STATE_NORM_CAP = 10.0  # max per-vector norm for any carried state tensor


def clip_state_norms(state: FullState) -> FullState:
    """Hard-cap per-vector norms in all carried state tensors.

    Prevents unbounded accumulation from EMA-style state updates (Z_narr stub,
    Z_eps, Z_homeo, etc.) compounding across thousands of steps.  Tensors whose
    norm is already ≤ _STATE_NORM_CAP are untouched; larger tensors are rescaled
    to exactly _STATE_NORM_CAP.  Non-tensor fields (ints, lists, None) pass through.
    Z_ssm (List[Optional[Tensor]]) is handled entry-by-entry: each SSM hidden-state
    tensor [B, n_heads, d_state, d_head] is per-vector-clamped along the last dim.
    Without this, the Mamba-3 SSM state can accumulate across steps with no bound,
    causing the kernel to produce NaN/Inf on step 3+ of training.
    """
    capped: dict = {}
    for k, v in vars(state).items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.is_floating_point():
            # Per-vector clipping along the last dimension
            norms = v.norm(dim=-1, keepdim=True)
            scale = (_STATE_NORM_CAP / norms.clamp(min=1e-6)).clamp(max=1.0)
            capped[k] = v * scale
        elif k == "Z_ssm" and v is not None:
            # Clamp each layer's SSM hidden state [B, n_heads, d_state, d_head] per last-dim vector.
            clamped_ssm = []
            for t in v:
                if isinstance(t, torch.Tensor) and t.is_floating_point() and t.ndim >= 1:
                    norms = t.norm(dim=-1, keepdim=True)
                    scale = (_STATE_NORM_CAP / norms.clamp(min=1e-6)).clamp(max=1.0)
                    clamped_ssm.append(t * scale)
                else:
                    clamped_ssm.append(t)
            capped[k] = clamped_ssm
        else:
            capped[k] = v
    return FullState(**capped)


def sanitize_state(state: FullState) -> FullState:
    """Replace NaN/Inf with 0 in all state tensors so NaN cannot leak between steps.

    clip_state_norms caps large finite values but passes NaN unchanged.  If any
    module produces NaN in a forward pass (e.g. Mamba kernel, controller entropy),
    that NaN lands in the returned state and poisons the next step's forward before
    any norm clamp can help.  sanitize_state is the last line of defense.
    """
    clean: dict = {}
    for k, v in vars(state).items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            clean[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        elif k == "Z_ssm" and v is not None:
            clean[k] = [
                torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                if isinstance(t, torch.Tensor) and t.is_floating_point()
                else t
                for t in v
            ]
        else:
            clean[k] = v
    return FullState(**clean)


def gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        norms[name] = float(param.grad.detach().norm().item())
    return norms


def load_val_chunks(tokenizer, seq_len: int, n_val: int) -> list:
    """Pre-tokenise wikitext-103 test split into fixed-length chunks for validation."""
    ds = load_dataset(
        "wikitext", "wikitext-103-raw-v1",
        split="test", streaming=True, trust_remote_code=False,
    )
    chunks, buf = [], []
    for ex in ds:
        if len(chunks) >= n_val:
            break
        text = ex["text"].strip()
        if not text:
            continue
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(buf) >= seq_len and len(chunks) < n_val:
            chunks.append(torch.tensor(buf[:seq_len], dtype=torch.long))
            buf = buf[seq_len:]
    return chunks


def evaluate(model, teacher, val_chunks: list, device: torch.device,
             temperature: float, amp_dtype: torch.dtype) -> float:
    """Mean KL distillation loss on held-out val set. Fresh state per chunk, no grad."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for chunk in val_chunks:
            input_ids = chunk.unsqueeze(0).to(device)
            teacher_logits = teacher(input_ids=input_ids).logits
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                state = model.init_state(batch_size=1)
                out = model(input_ids, state=state, step=0)
            if out.logits is None:
                continue
            loss = kl_distill_loss(
                out.logits.float(),
                teacher_logits[:, -1, :].float(),
                T=temperature,
            )
            total += loss.item()
    model.train()
    return total / max(len(val_chunks), 1)


def _diag_summary(outputs: ForwardOutputs) -> str:
    """One-line diagnostic string printed at eval steps."""
    d = outputs.diagnostics
    parts = []
    if "gamma_eff" in d:
        parts.append(f"γ_eff={d['gamma_eff'].mean().item():.3f}")
    if "mu_val" in d:
        parts.append(f"μ_val={d['mu_val'].mean().item():.3f}")
    if "v_self" in d:
        parts.append(f"V_self={d['v_self'].mean().item():.3f}")
    if "coherence" in d:
        parts.append(f"coh={d['coherence'].mean().item():.3f}")
    if "boredom" in d:
        parts.append(f"bore={d['boredom'].mean().item():.3f}")
    return "  ".join(parts)


def checkpoint_payload(
    *,
    args,
    step: int,
    model,
    optimizer,
    scheduler=None,
    le_state: FullState,
    best_loss: float,
    val_log: dict | None = None,
    ema_loss: float | None = None,
    event_manifest: dict | None = None,
) -> dict:
    return {
        "step": step,
        "variant": args.variant,
        "args": vars(args),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "le_state": _serialize_le_state(le_state),
        "best_loss": best_loss,
        "val_log": val_log if val_log is not None else {},
        "ema_loss": ema_loss,
        "event_manifest": event_manifest,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }


def save_checkpoint(path: str | Path, payload: dict) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return str(target)


def parse_args():
    p = argparse.ArgumentParser(description="V3.8 distillation with full telemetry")
    p.add_argument("--variant", default="round3_full")
    p.add_argument("--teacher", default="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled")
    p.add_argument("--tokenizer", default="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--d-state", type=int, default=128)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--telemetry-dir", default="telemetry/round3")
    p.add_argument("--print-full-json", action="store_true")
    p.add_argument("--out", default="checkpoints/round3_le_v35.pt")
    p.add_argument("--best-out", default="checkpoints/round3_le_v35_best.pt")
    p.add_argument("--forensic-dir", default="forensics/round3")
    p.add_argument("--pre-event-steps", type=int, default=128)
    p.add_argument("--post-event-steps-warn", type=int, default=32)
    p.add_argument("--post-event-steps-critical", type=int, default=64)
    p.add_argument("--baseline-window", type=int, default=100)
    p.add_argument("--forensic-cooldown", type=int, default=50)
    p.add_argument("--forensic-controller-cooldown", type=int, default=200,
                   help="Cooldown steps between controller-instability forensic bundles. "
                        "Higher than --forensic-cooldown to cap bundle count during action collapse.")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint .pt file to resume training from")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Save a periodic checkpoint every N steps (0 = final only)")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="LR warmup steps")
    p.add_argument("--state-store-dir", default=None,
                   help="Directory for StateStore disk persistence")
    p.add_argument("--consolidate-every", type=int, default=0,
                   help="Run a consolidating=True pass every N steps (0=disable)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Sequences per forward pass (A100 80GB: up to 8)")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--eval-every", type=int, default=0,
                   help="Evaluate on held-out val set every N steps (0=disable)")
    p.add_argument("--n-val", type=int, default=100,
                   help="Number of val chunks to pre-tokenise")
    def _maturity_window(v: str) -> int:
        n = int(v)
        if n < 100_000:
            raise argparse.ArgumentTypeError(f"--maturity-window must be >= 100,000 (got {n})")
        return n
    p.add_argument("--maturity-window", type=_maturity_window, default=100_000,
                   help="Rolling window for maturity gate and all metric trackers (minimum 100,000)")
    p.add_argument("--ab-eval-every", type=int, default=0,
                   help="Run multi-step A/B rollout eval every N steps (0=disable)")
    p.add_argument("--reload-test-every", type=int, default=0,
                   help="Run reload convergence test every N steps (0=disable)")
    p.add_argument("--ab-rollout-steps", type=int, default=20,
                   help="Val chunks per A/B rollout (controller + memory comparison)")
    p.add_argument("--snapshot-identity", action="store_true",
                   help="At end of training, commit I_0 = Z_id.clone().detach() and resave the "
                        "final checkpoint. Use after cycle3_identity to bake the formed identity "
                        "anchor into the checkpoint before advancing to integrated phases. "
                        "I_0=zeros means has_seed=0 in V_self (drift penalty suppressed); "
                        "seeding it here makes drift tracking live in round3_full and beyond.")
    p.add_argument("--use-8bit-adam", action="store_true",
                   help="Use bitsandbytes AdamW8bit instead of torch AdamW. "
                        "Saves ~56 GB of optimizer state for 7B student (float32 → uint8 moments). "
                        "Strongly recommended with Qwen3.6-35B-A3B teacher. "
                        "Requires: pip install bitsandbytes")
    p.add_argument("--teacher-4bit", action=argparse.BooleanOptionalAction, default=True,
                   help="Load teacher in 4-bit NF4 quantization (bitsandbytes). "
                        "Required for 35B teacher on A100 80 GB: bf16 alone consumes 70 GB. "
                        "Disable only if using a small teacher that fits in bf16. "
                        "Requires: pip install bitsandbytes")
    p.add_argument("--teacher-cpu", action="store_true", default=False,
                   help="Load teacher in bfloat16 on CPU instead of GPU. Frees ~22 GB VRAM "
                        "for the student at the cost of slower teacher inference per step. "
                        "Requires ~54 GB system RAM for a 27B teacher. "
                        "Takes precedence over --teacher-4bit.")
    p.add_argument("--no-forensics", action="store_true",
                   help="Skip forensic bundle collection and writing each step. "
                        "Console loss output is unaffected. Use when forensic bundles are not needed.")
    p.add_argument("--dataset-mix", default="wiki", choices=list(DATASET_MIXES),
                   help="Training data mix. 'wiki': WikiText-103 only (default, original behaviour). "
                        "'diverse': 15%% WikiText + 35%% C4 + 35%% GitHub code + 15%% OpenWebMath. "
                        "Validation always uses WikiText-103 test split regardless of this flag.")
    return p.parse_args()


def main():
    args = parse_args()
    # Set before the CUDA allocator is initialized (first tensor-to-device call).
    # expandable_segments prevents fragmentation from causing spurious OOM when the
    # 27B teacher's dequantization passes fragment the allocator cache.
    # Both names: PyTorch ≤2.1 uses PYTORCH_CUDA_ALLOC_CONF; 2.2+ uses PYTORCH_ALLOC_CONF.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype()
    use_scaler = amp_dtype == torch.float16
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.best_out) or ".", exist_ok=True)
    Path(args.telemetry_dir).mkdir(parents=True, exist_ok=True)
    forensic = ForensicEventManager(
        args.forensic_dir,
        ForensicConfig(
            pre_event_steps=args.pre_event_steps,
            post_event_steps_warn=args.post_event_steps_warn,
            post_event_steps_critical=args.post_event_steps_critical,
            baseline_window=args.baseline_window,
            cooldown_steps=args.forensic_cooldown,
            controller_cooldown_steps=args.forensic_controller_cooldown,
        ),
    )

    print(f"Device: {device}")
    print(f"Variant: {args.variant}")
    print(f"Telemetry dir: {args.telemetry_dir}")
    print(f"Forensic dir: {args.forensic_dir}")

    # Load tokenizer first so we can size the student's embedding/output_head correctly.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_vocab_size = len(tokenizer)
    print(f"Tokenizer: {args.tokenizer} — vocab_size={tokenizer_vocab_size}")

    config = build_config(args.variant)
    config = replace(config, d_state=args.d_state, device=str(device), vocab_size=tokenizer_vocab_size)
    student = build_model(args.variant, config, state_store_dir=args.state_store_dir).to(device)
    student.train()

    # Pre-warm the Triton backward autotuner BEFORE loading the teacher.
    # With only the student in VRAM (~14 GB), the autotuner has ~60+ GB of headroom
    # instead of ~5-10 GB during the first real training step.  Cached kernel configs
    # are reused for the rest of training, so this one-time cost pays for itself.
    if device.type == "cuda":
        print(f"Pre-warming Triton backward kernels (seq_len={args.seq_len}, batch={args.batch_size})...")
        torch.cuda.empty_cache()
        try:
            _dummy_ids = torch.zeros(args.batch_size, args.seq_len, dtype=torch.long, device=device)
            _dummy_state = student.init_state(batch_size=args.batch_size)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _dummy_out = student(_dummy_ids, state=_dummy_state, step=0)
            _dummy_out.logits.sum().backward()
            del _dummy_ids, _dummy_state, _dummy_out  # free GPU tensors before teacher loads
        except Exception as e:
            print(f"  Pre-warm failed (non-fatal): {e}")
        student.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        print("  Triton warmup done.")

    if args.teacher_cpu:
        # Weights live in CPU RAM; accelerate's cpu_offload moves each layer to GPU
        # just-in-time for computation then back to CPU. Only ~1.5 GB of teacher
        # weights occupy VRAM at any moment instead of ~22 GB. CUDA-only ops
        # (causal_conv1d etc.) still run on GPU so no kernel errors.
        # Requires ~54 GB system RAM for a 27B bfloat16 model.
        from accelerate import cpu_offload as _accel_cpu_offload
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=torch.bfloat16
        )
        _accel_cpu_offload(teacher, execution_device=device, offload_buffers=True)
        teacher_device = device  # inputs stay on GPU; hooks handle weight transfers
        print(f"Teacher: {args.teacher} (bfloat16, CPU-offloaded via accelerate — ~1.5 GB GPU peak per layer)")
    elif args.teacher_4bit:
        if not _BNB_AVAILABLE:
            raise bitsandbytes_requirement_error("--teacher-4bit")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=amp_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, quantization_config=bnb_cfg, device_map={"": 0}
        )
        teacher_device = device
        print(f"Teacher: {args.teacher} (4-bit NF4, device_map={{0}})")
    else:
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=amp_dtype).to(device)
        teacher_device = device
        print(f"Teacher: {args.teacher} ({amp_dtype})")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    val_chunks: list = []
    need_val = args.eval_every > 0 or args.ab_eval_every > 0 or args.reload_test_every > 0
    if need_val:
        print(f"Pre-tokenising {args.n_val} val chunks (test split)...")
        val_chunks = load_val_chunks(tokenizer, args.seq_len, args.n_val)
        print(f"  Loaded {len(val_chunks)} val chunks.")

    runner = EvalRunner(
        maturity_window=args.maturity_window,
        ab_rollout_steps=args.ab_rollout_steps,
        temperature=args.temperature,
    )

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if args.use_8bit_adam:
        if not _BNB_AVAILABLE:
            raise bitsandbytes_requirement_error("--use-8bit-adam")
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.lr, weight_decay=0.01)
        print("Optimizer: AdamW8bit (bitsandbytes) — ~18 GB saved vs float32 Adam")
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
        print("Optimizer: AdamW (torch float32)")
    warmup_steps = min(args.warmup_steps, args.steps // 10)
    def lr_schedule(s):
        if warmup_steps > 0 and s < warmup_steps:
            return (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(args.steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    data_gen = make_token_chunks(tokenizer, args.seq_len, args.dataset_mix)

    def next_chunk():
        nonlocal data_gen
        try:
            return next(data_gen)
        except StopIteration:
            data_gen = make_token_chunks(tokenizer, args.seq_len, args.dataset_mix)
            return next(data_gen)

    le_state = student.init_state(batch_size=args.batch_size)
    best_loss = float("inf")
    last_completed_step = 0
    start_step = 0
    val_log: dict[int, float] = {}
    ema_loss: float | None = None
    ema_alpha = 0.95
    accum_steps = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_sd = ckpt["model_state"]
        # Expand any checkpoint tensors whose vocab dimension grew (embed.weight, output_head.weight).
        # New-token rows are small-random so they don't bias training; existing rows carry warm-init.
        model_sd = student.state_dict()
        for key, ckpt_val in list(ckpt_sd.items()):
            if key not in model_sd:
                continue
            model_shape = model_sd[key].shape
            ckpt_shape = ckpt_val.shape
            if ckpt_shape != model_shape and len(ckpt_shape) >= 1 and ckpt_shape[1:] == model_shape[1:] and ckpt_shape[0] < model_shape[0]:
                new_weight = torch.randn(model_shape, dtype=ckpt_val.dtype) * 0.02
                new_weight[:ckpt_shape[0]] = ckpt_val
                ckpt_sd[key] = new_weight
                print(f"  vocab expand: {key} {list(ckpt_shape)} → {list(model_shape)}")
        student.load_state_dict(ckpt_sd, strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            # load_state_dict maps everything to CPU; move optimizer moments back to device
            for param_state in optimizer.state.values():
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        param_state[k] = v.to(device)
        saved_le = ckpt.get("le_state", {})
        if saved_le:
            le_state = _deserialize_le_state(saved_le, device)
            le_state.manifest = []   # manifest is ephemeral; rebuilt fresh each run
        start_step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        val_log = {int(k): v for k, v in ckpt.get("val_log", {}).items()}
        ema_loss = ckpt.get("ema_loss", None)
        last_completed_step = start_step
        warm_init = "optimizer_state" not in ckpt
        print(f"{'Warm-init' if warm_init else 'Resumed'} from step {start_step}: {args.resume}")
        if "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])

    prev_layer_input = None
    prev_kl: torch.Tensor | None = None

    def flush_finished_events(finished_events: list[dict], checkpoint_path: str | None) -> None:
        for ctx in finished_events:
            event_dir = forensic.write_bundle(ctx, checkpoint_path=checkpoint_path)
            print(f"forensic bundle saved: {event_dir}")

    for step in range(start_step + 1, args.steps + 1):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pre_state = detach_state(le_state)
        pre_epi_index = le_state.epi_index
        input_ids = torch.stack([next_chunk() for _ in range(args.batch_size)]).to(device)
        with torch.no_grad():
            _t_ids = input_ids if teacher_device == device else input_ids.to(teacher_device)
            teacher_last = teacher(input_ids=_t_ids, use_cache=False).logits[:, -1, :].float().to(device)

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            outputs = student(
                input_ids,
                state=le_state,
                step=step,
                prev_layer_input=prev_layer_input,
                distill_loss=prev_kl,
            )

        runner.record_train_step(step, outputs, pre_epi_index)

        if outputs.state is None:
            manifest = {
                "event_type": "voluntary_end",
                "trigger_name": "voluntary_end",
                "severity": "critical",
                "step": step,
            }
            event_ckpt = save_checkpoint(
                Path(args.forensic_dir) / f"voluntary_end_step_{step:06d}.pt",
                checkpoint_payload(
                    args=args,
                    step=step,
                    model=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    le_state=pre_state,
                    best_loss=best_loss,
                    event_manifest=manifest,
                ),
            )
            if not args.no_forensics:
                vol_snapshot = build_full_forensic_snapshot(
                    step=step,
                    variant_name=args.variant,
                    input_ids=input_ids,
                    teacher_logits=teacher_last,
                    student_logits=outputs.logits.float() if outputs.logits is not None else None,
                    pre_state=pre_state,
                    post_state=outputs.state,
                    outputs=outputs,
                    total_loss=None,
                    kl_loss=None,
                    grad_norms=None,
                )
                forensic.start_event(manifest, vol_snapshot, post_steps_override=0)
                flush_finished_events(forensic.finalize_all(), event_ckpt)
            # Restore best weights + spawn successor from predecessor identity/values
            best_ckpt_path = Path(args.best_out)
            if best_ckpt_path.exists():
                recovery = torch.load(str(best_ckpt_path), map_location="cpu", weights_only=False)
                student.load_state_dict(recovery["model_state"])
                optimizer.load_state_dict(recovery["optimizer_state"])
                for param_state in optimizer.state.values():
                    for k, v in param_state.items():
                        if isinstance(v, torch.Tensor):
                            param_state[k] = v.to(device)
                best_le_state = _deserialize_le_state(recovery["le_state"], device)
                archive = student.store.voluntary_consolidation(best_le_state)
                le_state = student.store.spawn_successor(archive, batch_size=args.batch_size)
                print(f"step={step} | action=VOLUNTARY_END | weights restored (step {recovery['step']}, best_loss={best_loss:.4f}) | successor spawned")
            else:
                le_state = student.init_state(batch_size=args.batch_size)
                print(f"step={step} | action=VOLUNTARY_END | no best checkpoint — fresh state")
            prev_layer_input = None
            prev_kl = None
            continue

        le_state = sanitize_state(clip_state_norms(detach_state(outputs.state)))
        prev_layer_input = outputs.diagnostics.get("layer_input")
        kl = kl_distill_loss(outputs.logits.float(), teacher_last, T=args.temperature)
        prev_kl = kl.detach()
        total_loss = kl + outputs.losses["L_total"]
        if not torch.isfinite(total_loss):
            # --- NaN/Inf diagnostics: identify the guilty component ---
            print(f"  [NaN diag] teacher_last finite={torch.isfinite(teacher_last).all().item()}"
                  f"  max={teacher_last.abs().max().item():.3g}")
            print(f"  [NaN diag] logits finite={torch.isfinite(outputs.logits).all().item()}"
                  f"  max={outputs.logits.abs().max().item():.3g}")
            print(f"  [NaN diag] kl={kl.item():.6f}  finite={torch.isfinite(kl).item()}")
            for lk, lv in outputs.losses.items():
                try:
                    print(f"  [NaN diag] {lk}={lv.item():.6f}  finite={torch.isfinite(lv).item()}")
                except Exception:
                    print(f"  [NaN diag] {lk}=<error>")
            # Check which state tensors contain NaN/Inf
            for sk, sv in vars(outputs.state).items():
                if isinstance(sv, torch.Tensor):
                    if not sv.isfinite().all():
                        print(f"  [NaN diag] state.{sk} has NaN/Inf  max={sv.abs().max().item():.3g}")
                elif sk == "Z_ssm" and sv is not None:
                    for li, t in enumerate(sv):
                        if isinstance(t, torch.Tensor) and not t.isfinite().all():
                            print(f"  [NaN diag] state.Z_ssm[{li}] has NaN/Inf  max={t.abs().max().item():.3g}")
            # Check pre_state too — if NaN is already in the input state it came from a prior step
            for sk, sv in vars(pre_state).items():
                if isinstance(sv, torch.Tensor):
                    if not sv.isfinite().all():
                        print(f"  [NaN diag] pre_state.{sk} has NaN/Inf  max={sv.abs().max().item():.3g}")
                elif sk == "Z_ssm" and sv is not None:
                    for li, t in enumerate(sv):
                        if isinstance(t, torch.Tensor) and not t.isfinite().all():
                            print(f"  [NaN diag] pre_state.Z_ssm[{li}] has NaN/Inf  max={t.abs().max().item():.3g}")
            # Check diagnostics dict for intermediate NaN
            for dk, dv in outputs.diagnostics.items():
                if isinstance(dv, torch.Tensor) and dv.is_floating_point():
                    if not dv.isfinite().all():
                        print(f"  [NaN diag] diagnostics.{dk} has NaN/Inf  max={dv.abs().max().item():.3g}")
            # -----------------------------------------------------------
            manifest = {
                "event_type": "nonfinite_failure",
                "trigger_name": "nonfinite_loss",
                "severity": "fatal",
                "step": step,
            }
            event_ckpt = save_checkpoint(
                Path(args.forensic_dir) / f"nonfinite_step_{step:06d}.pt",
                checkpoint_payload(
                    args=args,
                    step=step,
                    model=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    le_state=pre_state,
                    best_loss=best_loss,
                    event_manifest=manifest,
                ),
            )
            if not args.no_forensics:
                nf_snapshot = build_full_forensic_snapshot(
                    step=step,
                    variant_name=args.variant,
                    input_ids=input_ids,
                    teacher_logits=teacher_last,
                    student_logits=outputs.logits.float(),
                    pre_state=pre_state,
                    post_state=outputs.state,
                    outputs=outputs,
                    total_loss=total_loss,
                    kl_loss=kl,
                    grad_norms=None,
                )
                forensic.start_event(manifest, nf_snapshot, post_steps_override=0)
                flush_finished_events(forensic.finalize_all(), event_ckpt)
            print(f"step={step} | non-finite loss, stopping")
            break

        if use_scaler:
            scaler.scale(total_loss / args.grad_accum).backward()
        else:
            (total_loss / args.grad_accum).backward()
        accum_steps += 1

        grad_snapshot: dict[str, float] = {}
        if accum_steps == args.grad_accum:
            if use_scaler:
                scaler.unscale_(optimizer)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            if not torch.isfinite(total_grad_norm):
                # Non-finite grad norm means the backward pass produced NaN/Inf (most
                # likely from use_reentrant=True checkpoint re-running the Mamba kernel
                # with non-deterministic bfloat16 reduction order).  Skip this optimizer
                # step, revert state to pre-step, and continue — do NOT stop training.
                print(f"step={step} | non-finite grad norm ({float(total_grad_norm):.4g}) — skipping update, reverting state")
                optimizer.zero_grad(set_to_none=True)
                if use_scaler:
                    scaler.update()
                le_state = pre_state
                prev_layer_input = None
                prev_kl = None
                accum_steps = 0
            else:
                grad_snapshot = gradient_norms(student)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum_steps = 0

        if args.consolidate_every > 0 and step % args.consolidate_every == 0:
            with torch.no_grad():
                consol_out = student(
                    input_ids,
                    state=le_state,
                    step=step,
                    consolidating=True,
                )
                if consol_out.state is not None:
                    le_state = clip_state_norms(detach_state(consol_out.state))

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            periodic_path = str(Path(args.out).with_suffix("")) + f"_step{step:06d}.pt"
            save_checkpoint(
                periodic_path,
                checkpoint_payload(
                    args=args,
                    step=step,
                    model=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    le_state=le_state,
                    best_loss=best_loss,
                    val_log=val_log,
                    ema_loss=ema_loss,
                ),
            )
            print(f"periodic checkpoint saved: {periodic_path}")

        if not args.no_forensics:
            replay_entry = build_lightweight_replay_entry(
                step=step,
                input_ids=input_ids,
                teacher_logits=teacher_last,
                student_logits=outputs.logits.float(),
                pre_state=pre_state,
                post_state=outputs.state,
                outputs=outputs,
                total_loss=total_loss,
                kl_loss=kl,
            )
            forensic.append_replay_entry(replay_entry)
            full_snapshot = build_full_forensic_snapshot(
                step=step,
                variant_name=args.variant,
                input_ids=input_ids,
                teacher_logits=teacher_last,
                student_logits=outputs.logits.float(),
                pre_state=pre_state,
                post_state=outputs.state,
                outputs=outputs,
                total_loss=total_loss,
                kl_loss=kl,
                grad_norms=grad_snapshot,
            )
        else:
            full_snapshot = None
        last_completed_step = step

        total_loss_value = float(total_loss.detach().item())
        kl_value = float(kl.detach().item())
        ema_loss = total_loss_value if ema_loss is None else ema_alpha * ema_loss + (1 - ema_alpha) * total_loss_value

        if args.eval_every > 0 and step % args.eval_every == 0 and val_chunks:
            val_loss = evaluate(student, teacher, val_chunks, device, args.temperature, amp_dtype)
            val_log[step] = val_loss
            lr_now = scheduler.get_last_lr()[0]
            diag = _diag_summary(outputs)
            print(f"step={step:6d} | val={val_loss:.4f} | ema={ema_loss:.4f} | lr={lr_now:.2e}" + (f" | {diag}" if diag else ""))
            runner.print_report(step)

        if args.ab_eval_every > 0 and step % args.ab_eval_every == 0 and val_chunks:
            ab_summary = runner.run_ab_eval(student, teacher, val_chunks, le_state, device, amp_dtype)
            ctrl_d = ab_summary.get("ctrl_ab_delta", float("nan"))
            kl_n = ab_summary.get("kl_normal", float("nan"))
            kl_nc = ab_summary.get("kl_no_ctrl", float("nan"))
            line = f"  A/B  ctrl_delta={ctrl_d:.4f}  kl_normal={kl_n:.4f}  kl_no_ctrl={kl_nc:.4f}"
            mem_d = ab_summary.get("mem_ab_delta")
            if mem_d is not None:
                line += f"  mem_delta={mem_d:.4f}"
            print(line)

        if args.reload_test_every > 0 and step % args.reload_test_every == 0 and val_chunks:
            reload_summary = runner.run_reload_test(
                student, teacher, val_chunks, le_state, device, amp_dtype, step=step
            )
            print(
                f"  reload  D_id_before={reload_summary['d_id_at_reload']:.4f}"
                f"  D_id_final={reload_summary['d_id_final']:.4f}"
                f"  converges={reload_summary['converges']}"
                f"  steps={reload_summary['n_steps_run']}"
            )

        triggered_events: list = []
        event_checkpoint_path: str | None = None
        if not args.no_forensics:
            triggered_events = forensic.evaluate_triggers(
                step=step,
                outputs=outputs,
                total_loss=total_loss_value,
                kl_loss=kl_value,
            )
            for manifest in triggered_events:
                manifest["step"] = step
                forensic.start_event(manifest, full_snapshot)

        if total_loss_value < best_loss:
            best_loss = total_loss_value
            best_manifest = {
                "event_type": "best_checkpoint",
                "trigger_name": "best_checkpoint",
                "severity": "critical",
                "step": step,
                "best_loss": best_loss,
            }
            best_path = save_checkpoint(
                args.best_out,
                checkpoint_payload(
                    args=args,
                    step=step,
                    model=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    le_state=le_state,
                    best_loss=best_loss,
                    val_log=val_log,
                    ema_loss=ema_loss,
                    event_manifest=best_manifest,
                ),
            )
            if not args.no_forensics:
                forensic.start_event(best_manifest, full_snapshot, post_steps_override=0)
            event_checkpoint_path = best_path

        if not args.no_forensics and full_snapshot is not None:
            flush_finished_events(forensic.append_post_step(full_snapshot), event_checkpoint_path)
        else:
            flush_finished_events([], event_checkpoint_path)

        if step % args.log_every == 0 or step == 1:
            snapshot = {
                "step": step,
                "variant_name": args.variant,
                "action": outputs.action,
                "loss_kl": kl_value,
                "loss_total_with_kl": total_loss_value,
                "ema_loss": ema_loss,
                "losses": {k: {"mean": float(v.detach().float().mean().item())} for k, v in outputs.losses.items()},
            }
            print(render_snapshot_summary(snapshot))
            if args.print_full_json:
                print(json.dumps({
                    "summary": snapshot,
                    "triggered_events": triggered_events,
                }, indent=2))

    # §identity snapshot: commit I_0 = Z_id at end of cycle3_identity.
    # I_0 is the frozen anchor for L_id and the V_self drift term. Without this,
    # I_0 stays all-zeros forever: has_seed=0 (no drift penalty), and L_id = gamma_eff*||Z_id||²
    # (pulls toward zero, fighting identity formation). Run cycle3_identity first, then
    # pass --snapshot-identity to bake the shaped Z_id into I_0 before advancing to
    # round3_full, phase5, cycle6, or cycle7 (all have enable_identity=True).
    if args.snapshot_identity:
        le_state.I_0 = le_state.Z_id.detach().clone()
        i0_norm = float(le_state.I_0.norm().item())
        print(f"Identity snapshot: I_0 ← Z_id  (norm={i0_norm:.4f})")
        if i0_norm < 1e-3:
            print("  WARNING: I_0 norm is near-zero — Z_id may not have been shaped yet. "
                  "Verify cycle3_identity completed successfully before relying on this snapshot.")

    final_path = save_checkpoint(
        args.out,
        checkpoint_payload(
            args=args,
            step=last_completed_step,
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            le_state=le_state,
            best_loss=best_loss,
            val_log=val_log,
            ema_loss=ema_loss,
            event_manifest={"event_type": "final_checkpoint", "trigger_name": "final_checkpoint", "severity": "warning", "step": last_completed_step},
        ),
    )
    if not args.no_forensics:
        flush_finished_events(forensic.finalize_all(), final_path)
    print(f"checkpoint saved: {final_path}")


if __name__ == "__main__":
    main()
