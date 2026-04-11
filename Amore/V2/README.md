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

Layout:
- `life_eq_v2/config.py`: locked architecture and training conventions (includes §0.6 autonomy params)
- `life_eq_v2/state.py`: full state vector, manifest entries, initialization (includes Z_values, alpha_0)
- `life_eq_v2/modules.py`: section-level computational modules (includes ValueDynamicsModule, ViabilityModule)
- `life_eq_v2/model.py`: backbone, five-phase forward pass, objective assembly (§31)
- `life_eq_v2/persistence.py`: save/load/reset/spawn_successor API from section `30`
- `life_eq_v2/spec_check.py`: machine-readable checks for locked ordering and v15 autonomy invariants
- `tests/test_spec_lock.py`: smoke tests for shape, persistence, and spec ordering

The code is intentionally spec-first rather than retrofitted to the earlier experiment harness.
