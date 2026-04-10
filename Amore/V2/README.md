# Life Equation Computational Spec v2

This folder contains a fresh rebuild of the project around `life equation computational spec v2.pdf`.

Design goals:
- map the implementation one-to-one to the spec sections
- keep all new work isolated under `V2/`
- preserve the locked signal-flow conventions from section `0.5`
- provide a complete forward pass matching section `29`

Layout:
- `life_eq_v2/config.py`: locked architecture and training conventions
- `life_eq_v2/state.py`: full state vector, manifest entries, initialization
- `life_eq_v2/modules.py`: section-level computational modules
- `life_eq_v2/model.py`: backbone, five-phase forward pass, objective assembly
- `life_eq_v2/persistence.py`: save/load/reset/compatibility API from section `28`
- `life_eq_v2/spec_check.py`: machine-readable checks for locked ordering
- `tests/test_spec_lock.py`: smoke tests for shape, persistence, and spec ordering

The code is intentionally spec-first rather than retrofitted to the earlier experiment harness.
