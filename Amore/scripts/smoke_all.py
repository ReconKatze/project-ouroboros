#!/usr/bin/env python3
"""
Project Amore — comprehensive smoke test suite.

Runs in two phases:
  Unit tests  (--unit-only): imports, shapes, metrics — CPU only, no teacher
  Integration (default):     + 5-step forward passes with student + teacher on GPU

Usage (Colab):
  !python scripts/smoke_all.py                          # full suite
  !python scripts/smoke_all.py --unit-only              # fast, no GPU required
  !python scripts/smoke_all.py --teacher Qwen/Qwen2.5-Coder-7B
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "V3"))

import torch

# ──────────────────────────────────────────────────────────────────────────── #
# Test runner                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

_results: list[tuple[str, bool | None, str]] = []


def check(name: str, fn) -> bool:
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  [PASS] {name}")
        return True
    except Exception as e:
        _results.append((name, False, str(e)))
        print(f"  [FAIL] {name}")
        print(f"         {e}")
        if os.environ.get("SMOKE_VERBOSE"):
            traceback.print_exc()
        return False


def skip(name: str, reason: str = "") -> None:
    _results.append((name, None, reason))
    label = f"  [SKIP] {name}"
    print(label + (f" ({reason})" if reason else ""))


def final_summary() -> int:
    passed  = sum(1 for _, ok, _ in _results if ok is True)
    failed  = sum(1 for _, ok, _ in _results if ok is False)
    skipped = sum(1 for _, ok, _ in _results if ok is None)
    print()
    print("=" * 62)
    print(f"  {passed} passed  |  {failed} failed  |  {skipped} skipped")
    print("=" * 62)
    if failed:
        print("\nFailed:")
        for name, ok, msg in _results:
            if ok is False:
                print(f"  - {name}: {msg}")
    return failed


# ──────────────────────────────────────────────────────────────────────────── #
# Unit test 1 — imports                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

def test_imports():
    from life_eq_v3 import (                        # noqa: F401
        LifeEquationConfig, VariantProfile,
        build_config, build_model,
        FullState, zero_state, StateStore,
    )
    from life_eq_v3.modules import (                # noqa: F401
        SelfDynamicsModel,
        NarrativeModule,
        SleepModule,
        DreamModule,
        TrustModule,
    )
    from chimera.evaluation.metrics import (          # noqa: F401
        IdentityDriftTracker, ControllerMetrics, MemoryMetrics,
        CContMetrics, PerceptionCouplingMetrics, SelfModelMetrics,
        EvaluationReport, build_evaluation_report,
    )
    from chimera.evaluation.runner import EvalRunner  # noqa: F401
    from life_eq_v3.maturity_gate import (            # noqa: F401
        MaturityGate, GateThresholds, MaturityReport,
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Unit test 2 — SelfDynamicsModel shapes + has_prev guard + lookahead          #
# ──────────────────────────────────────────────────────────────────────────── #

def test_sdm_shapes():
    from life_eq_v3 import build_config
    from life_eq_v3.modules import SelfDynamicsModel

    config = build_config("phase1_core_dynamics")
    sdm    = SelfDynamicsModel(config)
    B      = 2

    h0      = torch.zeros(B, config.d_sdm)
    pp_zero = torch.zeros(B, 4)           # simulates first step of lifetime
    pp_fill = torch.rand(B, 4)            # simulates step 2+
    summary = torch.rand(B, 4)
    act     = torch.zeros(B, dtype=torch.long)

    # Step 1: prev_pred is all-zero → has_prev = 0 → l_self must be 0
    pred, h, l = sdm(summary_t=summary, prev_pred=pp_zero, action_idx=act, h_prev=h0)
    assert pred.shape == (B, 4),           f"pred shape mismatch: {pred.shape}"
    assert h.shape    == (B, config.d_sdm), f"h shape mismatch: {h.shape}"
    assert l.item()   == 0.0,              f"l_self should be 0 on first step, got {l.item()}"

    # Step 2: non-zero prev_pred → has_prev = 1 → l_self > 0
    _, _, l2 = sdm(summary_t=summary, prev_pred=pp_fill, action_idx=act, h_prev=h)
    assert l2.item() > 0.0, f"l_self should be >0 on step 2, got {l2.item()}"

    # Lookahead: k steps, all under no_grad, correct shapes
    preds = sdm.lookahead(summary[0:1], h0[0:1], k=5)
    assert len(preds) == 5,          f"lookahead returned {len(preds)} steps, want 5"
    assert preds[0].shape == (1, 4), f"lookahead step shape: {preds[0].shape}"


# ──────────────────────────────────────────────────────────────────────────── #
# Unit test 3 — FullState backward compat (old checkpoint without SDM fields)  #
# ──────────────────────────────────────────────────────────────────────────── #

def test_fullstate_backward_compat():
    from life_eq_v3 import build_config
    from life_eq_v3.state import zero_state, FullState

    config = build_config("phase1_core_dynamics")
    full   = zero_state(batch_size=1, config=config)

    # Simulate an old checkpoint: serialise to dict, drop the SDM keys
    state_dict = vars(full).copy()
    for key in ("Z_sdm", "Z_sdm_pred", "prev_action_idx"):
        state_dict.pop(key, None)

    # Must reconstruct without error (new fields have defaults)
    state = FullState(**state_dict)
    assert state.Z_sdm        is None, "Z_sdm should default to None"
    assert state.Z_sdm_pred   is None, "Z_sdm_pred should default to None"
    assert state.prev_action_idx == 0, "prev_action_idx should default to 0"


# ──────────────────────────────────────────────────────────────────────────── #
# Unit test 4 — SelfModelMetrics                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def test_self_model_metrics():
    from chimera.evaluation.metrics import SelfModelMetrics

    sm = SelfModelMetrics(window=20)
    for i in range(25):
        sm.record(
            l_self=0.1 * (1.0 - i / 25.0),
            pred=(float(i) * 0.01,) * 4,   # tuples of 4 floats
            actual=(0.0, 0.0, 0.0, 0.0),
            v_self_augmented=(i % 3 == 0),
        )
    s = sm.summary()
    assert "l_self_mean"          in s, "l_self_mean missing"
    assert "pessimism_active_rate" in s, "pessimism_active_rate missing"
    assert 0.0 <= s["pessimism_active_rate"] <= 1.0, \
        f"pessimism_active_rate out of range: {s['pessimism_active_rate']}"
    for dim in ("d_id", "eps_norm", "c_cont", "v_self"):
        assert f"mse_{dim}" in s, f"mse_{dim} missing from summary"


# ──────────────────────────────────────────────────────────────────────────── #
# Unit test 5 — EvalRunner report (no model needed)                            #
# ──────────────────────────────────────────────────────────────────────────── #

def test_eval_runner_report():
    from chimera.evaluation.runner import EvalRunner

    runner = EvalRunner(maturity_window=50, ab_rollout_steps=5, temperature=2.0)
    for i in range(60):
        runner.self_model.record(l_self=0.05 + 0.01 * (i % 5))

    mr, er = runner.report(step=60)
    # Both reports must serialise without error
    _ = mr.summary()
    _ = er.summary()
    assert hasattr(er, "self_model"), "EvaluationReport missing self_model field"


# ──────────────────────────────────────────────────────────────────────────── #
# Unit test 6 — NarrativeModule shapes + EMA correctness                        #
# ──────────────────────────────────────────────────────────────────────────── #

def test_narrative_module():
    from life_eq_v3 import build_config
    from life_eq_v3.modules import NarrativeModule

    config = build_config("phase1_core_dynamics")
    module = NarrativeModule(config)
    module.eval()
    B = 2

    z_narr    = torch.zeros(B, config.d_narr)
    z_auto    = torch.zeros(B, config.d_auto)
    late_pool = torch.randn(B, config.d_state)
    active_id = torch.randn(B, config.d_state)
    dream_res = torch.randn(B, config.d_narr)

    # Awake step: z_narr updates via GRU; z_auto EMA moves toward target
    with torch.no_grad():
        z_narr_out, z_auto_out = module(z_narr, z_auto, late_pool, active_id, None, False)

    assert z_narr_out.shape == (B, config.d_narr), f"z_narr shape: {z_narr_out.shape}"
    assert z_auto_out.shape == (B, config.d_auto), f"z_auto shape: {z_auto_out.shape}"
    # GRU output should differ from the zero initial state
    assert not torch.allclose(z_narr_out, z_narr), "z_narr should change after GRU step"

    # Z_auto EMA: verify the update equation directly
    with torch.no_grad():
        target = module.id_to_auto(active_id)                                   # [B, d_narr]
    tau = config.tau_auto
    expected_auto = z_auto[:, :config.d_narr] + (target - z_auto[:, :config.d_narr]) / tau
    assert torch.allclose(z_auto_out[:, :config.d_narr], expected_auto, atol=1e-5), \
        "Z_auto EMA update does not match expected"

    # Consolidating with dream residual: z_narr must differ from the awake update
    with torch.no_grad():
        z_narr_cons, _ = module(z_narr, z_auto, late_pool, active_id, dream_res, True)
    assert z_narr_cons.shape == (B, config.d_narr)
    assert not torch.allclose(z_narr_cons, z_narr_out), \
        "consolidating z_narr should differ from awake z_narr (dream residual has no effect)"


# ──────────────────────────────────────────────────────────────────────────── #
# Unit test 7 — SleepModule: accumulate awake, release consolidating            #
# ──────────────────────────────────────────────────────────────────────────── #

def test_sleep_module():
    from life_eq_v3 import build_config
    from life_eq_v3.modules import SleepModule

    config = build_config("phase1_core_dynamics")
    module = SleepModule(config)
    module.eval()
    B = 2

    z_sleep = torch.zeros(B, 1)
    z_att   = torch.randn(B, config.d_att)
    z_eps   = torch.randn(B, config.d_eps)
    z_pfat  = torch.ones(B, 1) * 2.0

    # Awake: pressure should increase (or at minimum stay ≥ 0)
    with torch.no_grad():
        z_sleep_out = module(z_sleep, z_att, z_eps, z_pfat, consolidating=False)
    assert z_sleep_out.shape == (B, 1), f"z_sleep shape: {z_sleep_out.shape}"
    assert (z_sleep_out >= z_sleep).all(), \
        "sleep pressure should not decrease when awake"
    assert (z_sleep_out >= 0.0).all(), "sleep pressure must be ≥ 0"

    # Consolidating: pressure releases toward zero
    z_sleep_high = torch.ones(B, 1) * 2.0
    with torch.no_grad():
        z_sleep_dec = module(z_sleep_high, z_att, z_eps, z_pfat, consolidating=True)
    assert (z_sleep_dec < z_sleep_high).all(), \
        "sleep pressure should decrease during consolidation"
    assert (z_sleep_dec >= 0.0).all(), "sleep pressure must not go negative"
    assert (z_sleep_dec <= 10.0).all(), "sleep pressure must not exceed max"


# ──────────────────────────────────────────────────────────────────────────── #
# Unit test 8 — DreamModule: fade awake, replay consolidating                   #
# ──────────────────────────────────────────────────────────────────────────── #

def test_dream_module():
    from life_eq_v3 import build_config, zero_state
    from life_eq_v3.modules import DreamModule

    config = build_config("phase1_core_dynamics")
    module = DreamModule(config)
    module.eval()
    B = 1

    z_dream = torch.ones(B, config.d_dream)
    state   = zero_state(batch_size=B, config=config)

    # Awake: z_dream fades, no narrative residual
    with torch.no_grad():
        z_dream_awake, narr_awake = module(z_dream, state, consolidating=False)
    assert z_dream_awake.shape == (B, config.d_dream), f"awake z_dream shape: {z_dream_awake.shape}"
    assert narr_awake is None, "narr_residual must be None when awake"
    assert (z_dream_awake.abs() <= z_dream.abs() + 1e-6).all(), \
        "z_dream should fade (magnitude ≤ input) when awake"

    # Consolidating with empty episodic buffer: z_dream unchanged, no residual
    with torch.no_grad():
        z_dream_empty, narr_empty = module(z_dream, state, consolidating=True)
    assert narr_empty is None, "narr_residual must be None when epi buffer empty"

    # Consolidating with filled episodic buffer: z_dream updates, residual returned
    state.epi_vals[:4] = torch.randn(4, config.d_val)
    state.epi_index    = 4
    with torch.no_grad():
        z_dream_cons, narr_cons = module(z_dream, state, consolidating=True)
    assert z_dream_cons.shape == (B, config.d_dream), f"consolidating z_dream shape"
    assert narr_cons is not None, \
        "narr_residual must be non-None during consolidation with filled buffer"
    assert narr_cons.shape == (B, config.d_narr), f"narr_residual shape: {narr_cons.shape}"


# ──────────────────────────────────────────────────────────────────────────── #
# Unit test 9 — TrustModule: bounded output, direction under adversarial state  #
# ──────────────────────────────────────────────────────────────────────────── #

def test_trust_module():
    from life_eq_v3 import build_config
    from life_eq_v3.modules import TrustModule

    config = build_config("phase1_core_dynamics")
    module = TrustModule(config)
    module.eval()
    B = 2

    t_trust   = torch.tensor([[config.trust_default]])   # [1, 1]
    z_eps     = torch.zeros(B, config.d_eps)              # low error → high reliability
    coherence = torch.ones(B, 1)                          # perfect coherence
    d_id      = torch.zeros(B, 1)                         # zero drift → stable

    with torch.no_grad():
        t_new = module(t_trust, z_eps, coherence, d_id)

    assert t_new.shape == t_trust.shape, f"T_trust shape: {t_new.shape}"
    assert (t_new >= 0.0).all() and (t_new <= 1.0).all(), \
        f"T_trust must be in [0, 1], got {t_new.item():.4f}"

    # Under adversarial state: trust target < 1 → T_trust should not increase from 1.0
    z_eps_bad     = torch.ones(B, config.d_eps) * 5.0
    coherence_bad = -torch.ones(B, 1)
    d_id_bad      = torch.ones(B, 1) * 3.0
    t_trust_high  = torch.ones(1, 1)

    with torch.no_grad():
        t_adversarial = module(t_trust_high, z_eps_bad, coherence_bad, d_id_bad)
    assert (t_adversarial <= t_trust_high + 1e-6).all(), \
        f"Trust should not increase under adversarial state, got {t_adversarial.item():.4f}"
    assert (t_adversarial >= 0.0).all(), "T_trust must never go negative"


# ──────────────────────────────────────────────────────────────────────────── #
# Integration test 10 — 5-step forward, round3 modules enabled                 #
# ──────────────────────────────────────────────────────────────────────────── #

def test_integration_round3(teacher, device, amp_dtype):
    """Verify NarrativeModule, SleepModule/DreamModule, TrustModule fire
    without error and produce sensible state updates."""
    import torch.nn.functional as F
    from life_eq_v3.factory import build_config, build_model
    from life_eq_v3.state import FullState
    from dataclasses import replace as dc_replace

    config  = build_config("round3_narrative_sleep_trust")
    config  = dc_replace(config, d_state=128, device=str(device))
    student = build_model("round3_narrative_sleep_trust", config).to(device)
    student.train()

    le_state  = student.init_state(batch_size=1)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=5e-5
    )

    z_narr_initial  = le_state.Z_narr.clone()
    z_sleep_initial = le_state.Z_sleep.clone()
    t_trust_initial = le_state.T_trust.clone()

    for step in range(5):
        input_ids = torch.randint(0, 151936, (1, 64), device=device)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids).logits[:, -1, :].float()

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = student(input_ids, state=le_state, step=step)

        if out.logits is None:
            continue  # VOLUNTARY_END — skip

        assert out.losses is not None, f"losses is None at step {step}"
        assert not torch.isnan(out.logits).any(), f"NaN in logits at step {step}"

        # Backward pass
        vocab = min(out.logits.size(-1), teacher_logits.size(-1))
        kl = F.kl_div(
            F.log_softmax(out.logits[..., :vocab].float() / 2.0, dim=-1),
            F.softmax(teacher_logits[..., :vocab].float() / 2.0, dim=-1),
            reduction="batchmean",
        ) * 4.0
        total = kl + out.losses.get("L_total", torch.tensor(0.0, device=device))
        total.backward()
        optimizer.step()
        optimizer.zero_grad()

        le_state = FullState(**{
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in vars(le_state).items()
        })

    # After 5 steps the new state fields must have updated and stayed bounded
    assert not torch.allclose(le_state.Z_narr, z_narr_initial.to(device)), \
        "Z_narr should update across steps (NarrativeModule not firing?)"
    assert (le_state.Z_sleep >= 0.0).all(), "Z_sleep went negative"
    assert (le_state.T_trust >= 0.0).all() and (le_state.T_trust <= 1.0).all(), \
        f"T_trust out of [0,1]: {le_state.T_trust}"

    del student, optimizer


# ──────────────────────────────────────────────────────────────────────────── #
# Integration test 11 — 5-step forward, SDM disabled (regression baseline)    #
# ──────────────────────────────────────────────────────────────────────────── #

def test_integration_no_sdm(teacher, device, amp_dtype):
    from life_eq_v3.factory import build_config, build_model
    from life_eq_v3.state import FullState
    from dataclasses import replace

    config  = build_config("phase1_core_dynamics")
    config  = replace(config, d_state=128, device=str(device))
    student = build_model("phase1_core_dynamics", config).to(device)
    student.train()

    le_state = student.init_state(batch_size=1)

    for step in range(5):
        input_ids = torch.randint(0, 151936, (1, 64), device=device)
        with torch.no_grad():
            _ = teacher(input_ids=input_ids).logits  # warm teacher cache

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = student(input_ids, state=le_state, step=step)

        if out.logits is None:
            continue  # VOLUNTARY_END — skip this step

        assert out.losses is not None, "losses is None"
        l_sm = out.losses.get("L_self_model", torch.tensor(0.0))
        assert l_sm.item() == 0.0, \
            f"L_self_model should be 0 with SDM disabled, got {l_sm.item()}"
        assert not torch.isnan(out.logits).any(), "NaN in logits"

        le_state = FullState(**{
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in vars(le_state).items()
        })

    del student


# ──────────────────────────────────────────────────────────────────────────── #
# Integration test 12 — 5-step forward, SDM enabled (patched profile)         #
# ──────────────────────────────────────────────────────────────────────────── #

def test_integration_with_sdm(teacher, device, amp_dtype):
    import torch.nn.functional as F
    from life_eq_v3.factory import build_config, build_model
    from life_eq_v3.state import FullState
    from dataclasses import replace as dc_replace

    config  = build_config("phase1_core_dynamics")
    config  = dc_replace(config, d_state=128, device=str(device))
    student = build_model("phase1_core_dynamics", config).to(device)
    object.__setattr__(
        student, "profile",
        dc_replace(student.profile, enable_self_dynamics=True),
    )
    student.train()

    le_state  = student.init_state(batch_size=1)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=5e-5
    )

    for step in range(5):
        input_ids = torch.randint(0, 151936, (1, 64), device=device)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids).logits[:, -1, :].float()

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = student(input_ids, state=le_state, step=step)

        if out.logits is None:
            continue

        # L_self_model must be present and finite
        assert "L_self_model" in out.losses, \
            f"L_self_model missing from losses at step {step}"
        l_sm = out.losses["L_self_model"]
        assert not torch.isnan(l_sm), f"L_self_model is NaN at step {step}"

        # has_prev guard: step 0 pred was all-zero → l_self must be 0
        if step == 0:
            assert l_sm.item() == 0.0, \
                f"step 0 l_self should be 0 (has_prev guard), got {l_sm.item()}"

        # SDM diagnostic must be emitted
        diag = out.diagnostics or {}
        assert "sdm_l_self" in diag, \
            f"sdm_l_self missing from diagnostics at step {step}"

        # SDM state fields must be populated
        assert le_state.Z_sdm      is not None, "Z_sdm not updated after forward"
        assert le_state.Z_sdm_pred is not None, "Z_sdm_pred not updated after forward"
        assert le_state.Z_sdm.shape[-1] == config.d_sdm, \
            f"Z_sdm dim mismatch: {le_state.Z_sdm.shape[-1]} vs {config.d_sdm}"

        # Backward pass must not explode
        vocab = min(out.logits.size(-1), teacher_logits.size(-1))
        kl = F.kl_div(
            F.log_softmax(out.logits[..., :vocab].float() / 2.0, dim=-1),
            F.softmax(teacher_logits[..., :vocab].float() / 2.0, dim=-1),
            reduction="batchmean",
        ) * 4.0
        total = kl + out.losses.get("L_total", torch.tensor(0.0, device=device))
        total.backward()
        optimizer.step()
        optimizer.zero_grad()

        le_state = FullState(**{
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in vars(le_state).items()
        })

    del student, optimizer


# ──────────────────────────────────────────────────────────────────────────── #
# Training command printed on success                                           #
# ──────────────────────────────────────────────────────────────────────────── #

TRAINING_CMD = textwrap.dedent("""\
    python scripts/train_distill_v3.py \\
        --variant phase4_infrastructure \\
        --teacher Qwen/Qwen2.5-Coder-7B \\
        --tokenizer Qwen/Qwen2.5-Coder-1.5B \\
        --steps 10000 \\
        --d-state 128 \\
        --batch-size 4 \\
        --grad-accum 2 \\
        --lr 5e-5 \\
        --warmup-steps 200 \\
        --seq-len 1024 \\
        --eval-every 500 \\
        --n-val 100 \\
        --ab-eval-every 1000 \\
        --ab-rollout-steps 20 \\
        --reload-test-every 2000 \\
        --maturity-window 10000 \\
        --checkpoint-every 1000 \\
        --out checkpoints/step4_le_v3.pt \\
        --best-out checkpoints/step4_le_v3_best.pt \\
        --telemetry-dir telemetry/step4 \\
        --forensic-dir forensics/step4

    # To resume from a checkpoint, append:
    #   --resume checkpoints/step4_le_v3.pt
""")


# ──────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(description="Project Amore smoke test suite")
    p.add_argument(
        "--unit-only", action="store_true",
        help="Skip GPU integration tests (no teacher model required)",
    )
    p.add_argument(
        "--teacher", default="Qwen/Qwen2.5-Coder-7B",
        help="HuggingFace ID of the teacher model",
    )
    p.add_argument(
        "--tokenizer-id", default="Qwen/Qwen2.5-Coder-1.5B",
        help="HuggingFace ID of the tokenizer",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print()
    print("=" * 62)
    print("  Project Amore — smoke test suite")
    print("=" * 62)

    # ── Unit tests (CPU, no teacher) ──────────────────────────────────────
    print("\n[Unit tests — CPU]\n")
    check("imports",                              test_imports)
    check("SDM: shapes + has_prev + lookahead",   test_sdm_shapes)
    check("FullState: backward compat",           test_fullstate_backward_compat)
    check("SelfModelMetrics",                     test_self_model_metrics)
    check("EvalRunner: report (no model)",        test_eval_runner_report)
    check("NarrativeModule: shapes + EMA",        test_narrative_module)
    check("SleepModule: accumulate + release",    test_sleep_module)
    check("DreamModule: fade + replay",           test_dream_module)
    check("TrustModule: bounded + direction",     test_trust_module)

    # ── Integration tests (GPU + teacher) ────────────────────────────────
    print("\n[Integration tests — GPU + teacher]\n")

    if args.unit_only:
        skip("5-step forward, round3 modules",  "--unit-only")
        skip("5-step forward, SDM off",         "--unit-only")
        skip("5-step forward, SDM on",          "--unit-only")

    elif not torch.cuda.is_available():
        skip("5-step forward, round3 modules",  "no CUDA device")
        skip("5-step forward, SDM off",         "no CUDA device")
        skip("5-step forward, SDM on",          "no CUDA device")

    else:
        from transformers import AutoModelForCausalLM

        device    = torch.device("cuda")
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        print(f"  Loading teacher: {args.teacher}")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=amp_dtype
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"  Teacher loaded ({sum(p.numel() for p in teacher.parameters()) / 1e9:.1f}B params)")
        print()

        check(
            "5-step forward, round3 modules (narrative/sleep/trust)",
            lambda: test_integration_round3(teacher, device, amp_dtype),
        )
        check(
            "5-step forward, SDM off (regression baseline)",
            lambda: test_integration_no_sdm(teacher, device, amp_dtype),
        )
        check(
            "5-step forward, SDM on  (patched profile)",
            lambda: test_integration_with_sdm(teacher, device, amp_dtype),
        )

        del teacher
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────
    failed = final_summary()

    if not failed:
        print("\nAll tests passed. Training command:\n")
        print(TRAINING_CMD)

    raise SystemExit(failed)


if __name__ == "__main__":
    main()
