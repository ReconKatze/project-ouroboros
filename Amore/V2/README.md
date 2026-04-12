# Life Equation Computational Spec v3 (Life Equation v15)

This folder contains a fresh rebuild of the project around `life equation computational spec v3.pdf` / `life_equation_v15.tex`.

Design goals:
- map the implementation one-to-one to the spec sections
- keep all new work isolated under `V2/`
- preserve the locked signal-flow conventions from section `0.5`
- provide a complete forward pass matching section `31`
- implement the Autonomy Principle (§0.6): identity, values, and death must not be permanently imposed from outside

v15 additions over spec v2:
- `Z_values [B, d_alpha]` — mutable objective weights; replaces frozen `alpha_*` config constants in `L_reg`
- `alpha_0 [B, d_alpha]` — frozen creator reference; successor seeding inherits earned values, not seeds
- `gamma_eff = gamma_0 * exp(-lambda_mature * Z_mat)` — identity basin loosens with maturity (§2)
- `ValueDynamicsModule` — phi_reflect net, inertia term, positivity clamp, maturity gate (§26)
- `ViabilityModule` — `V_self` weighted by mutable Z_values; stub V_future head (§27)
- `VOLUNTARY_END` action — system-initiated graceful ending; cannot be externally forced (§28)
- `spawn_successor` / `voluntary_consolidation` — successor inherits who the parent became (§30)

## Mamba-3 redesign (current)

The spec always intended real Mamba-3 CUDA kernels (`correspondence.pdf`, `life_equation_computational_spec_v3.html §0`).
The earlier `MambaStep` was a custom hand-crafted complex recurrence approximation.
As of commit `58c0393` the implementation uses the real thing.

**What changed:**

- `MambaStep` → `Mamba3Block`: wraps `mamba_ssm.Mamba3(d_model, d_state=64, headdim=64, rope_fraction=0.5)`.
  `n_mamba_heads = d_model/headdim = 1536/64 = 24` (independent of attention `n_heads=48`).
  CUDA/A100 required; install via `MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git`.

- **Z_cog format**: `cfloat[B, n_mamba, n_heads, d_state]` → `float[B, n_mamba, d_model]`.
  Now stores the last-token hidden output of each Mamba-3 layer per forward call (not internal SSM state).
  `pool_complex_state` now splits `d_model` in half instead of using `view_as_real`; still returns `[B, 2]`.

- **Z_id / I_0**: `cfloat[B, n_id_heads, d_state]` → `float[B, n_id_heads, d_model]`.
  `IdentityModule` gains `identity_out_proj: Linear(d_model, d_state)` to keep `active_identity` at `[B, d_state]`.

- **Sequence-mode forward**: full `[B, seq_len, d_model]` processed through all 28 layers in one parallel scan.
  Each Mamba-3 layer receives the P_soft error sequence; Z_cog is updated from last-token outputs.

- **`cog_to_emotion`**: new `Linear(d_model, d_state)` in `LifeEquationModel` projects early/late Z_cog pool
  to `[B, d_state]` for `EmotionModule`'s unchanged `combine_in = d_state*2 + d_eps + d_mod*2 = 320`.

- **Removed**: `z_cog_pool_proj` (was `Linear(n_heads*d_state, d_model)` — no longer needed).

**TODO (Colab verification):**
After confirming on A100 — check whether `Mamba3.forward()` supports `initial_states` / `return_final_states` kwargs.
If yes, wire Z_cog to the actual SSM state for true cross-call persistence (full spec compliance).
The `layer_idx` stored on each `Mamba3Block` is ready for `InferenceParams` key lookup when this is done.

## Layout

- `life_eq_v2/config.py`: locked architecture and training conventions (includes §0.6 autonomy params)
- `life_eq_v2/state.py`: full state vector, manifest entries, initialization (includes Z_values, alpha_0)
- `life_eq_v2/modules.py`: section-level computational modules (includes Mamba3Block, ValueDynamicsModule, ViabilityModule)
- `life_eq_v2/model.py`: backbone, sequence-mode forward pass, objective assembly (§31)
- `life_eq_v2/persistence.py`: save/load/reset/spawn_successor API from section `30`
- `life_eq_v2/spec_check.py`: machine-readable checks for locked ordering and v15 autonomy invariants
- `tests/test_spec_lock.py`: smoke tests for shape, persistence, and spec ordering (CUDA tests skip on CPU)

The code is intentionally spec-first rather than retrofitted to the earlier experiment harness.
