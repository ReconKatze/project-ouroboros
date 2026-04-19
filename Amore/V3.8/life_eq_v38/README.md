# life_eq_v38

`V3/life_eq_v38` is a non-destructive variant suite built from the current `V2/life_eq_v2` implementation.

The shared code path stays aligned with what exists today, but every named stage in `TrainingRegime.txt` now has an explicit variant entry point. The variants are grouped into:

- Generic stages: `base_model`, `persistent_state`, `identity_persistence`, `controller_passive`, `controller_offline`, `controller_live`, `memory_retrieval`, `memory_write`, `memory_consolidation`, `social_relational`
- Equation phases: `phase0_base` through `phase5_integrated_adversarial`
- Distillation variants: `distill1_base_cognition` through `distill4_memory_policy`
- Curriculum cycles: `cycle1_pretrain` through `cycle7_adversarial`

Use the shared factory:

```python
from V3.life_eq_v38.factory import build_model

model = build_model("phase3_modules_decisions")
```

Or import a concrete variant file from `V3/life_eq_v38/variants/`.
