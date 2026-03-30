# Project Ouroboros

Zero-cost validation pipeline for Project Chimera (~9.3B hybrid Mamba/Transformer coding model).
Proves every novel architectural concept at small scale (0.5B) before spending $1,150-$2,298 on Lambda.ai.

## Environment

- **Working directory**: `/run/media/deck/Katze2/Project Ouroboros/`
- **Python venv**: `.venv/` (Python 3.13, CPU-only torch 2.11, transformers 5.4)
- **Activate**: `source .venv/bin/activate`
- **HF cache**: Set `HF_HOME="/run/media/deck/Katze2/Project Ouroboros/.cache/huggingface"` before running scripts
- **Hardware**: Steam Deck (AMD GPU, no CUDA). CPU-only for Steps 1 & 5. Colab T4 for Steps 2-4.
- **Space constraint**: All downloads must go to this external drive, not the main drive.

## Project Structure

```
chimera/
  models/
    hybrid_model.py       # HybridChimeraModel, SinkTokens, MambaAttentionWrapper, GPTNeoXAdapter, Qwen2Adapter
    fallback_mamba.py     # FallbackMamba (CPU-only gated linear unit)
    real_mamba.py         # RealMambaBlock (mamba-ssm, CUDA), create_mamba_block() factory
  utils/
    weight_conversion.py  # extract_qkv(), tile_gqa_weights(), extract_qkv_pythia/separate()
    layer_plan.py         # build_layer_plan(), ATTN_KEEP_DEFAULTS
scripts/
  convert_and_test.py     # Steps 1 & 2 entry point (model-agnostic)
  colab_setup.sh          # Colab T4 dependency installer
```

## Steps & Status

| Step | Description | Hardware | Status |
|------|-------------|----------|--------|
| 1 | Pythia-160M Mamba conversion | CPU | **COMPLETE** |
| 2 | Qwen2.5-0.5B conversion (real Mamba kernels) | Colab T4 | **COMPLETE** |
| 3 | Distillation from 3B teacher | Colab T4 | Not started |
| 4 | d_state gradient experiment (4 variants) | Colab T4 | Not started |
| 5 | GGUF export | CPU | Not started |

## Critical Technical Details Discovered

### Pythia QKV Format (Interleaved Per-Head)
GPT-NeoX uses interleaved `[Q0,K0,V0, Q1,K1,V1, ...]` format, NOT block `[QQQ...KKK...VVV...]`.
**Do NOT use `torch.chunk(weight, 3, dim=0)`** — this silently corrupts weights.
Correct split:
```python
w = weight.view(num_heads, 3, head_size, hidden_size)
q_w = w[:, 0, :, :].contiguous().view(-1, hidden_size)
k_w = w[:, 1, :, :].contiguous().view(-1, hidden_size)
v_w = w[:, 2, :, :].contiguous().view(-1, hidden_size)
```

### GPT-NeoX Layer Interface
- **Parallel residual**: `output = x + attention(norm(x)) + mlp(norm2(x))` — MLP is untouched when replacing attention
- **Layer norms input before calling attention** — the MambaAttentionWrapper must NOT re-norm
- **Attention returns 2-tuple**: `(output, present_kv)` — NOT 3-tuple
- **Layer returns tensor directly** — NOT a tuple
- **Rotary embeddings**: Must be computed via `model.gpt_neox.rotary_emb()` and passed as `position_embeddings=` to each layer
- **Causal mask**: Must be created via `create_causal_mask()` from transformers; if dtype is long, cast to float

### Weight Mapping (v1.2 Corrected)
- Q weights → `out_proj_c` (C = output read projection)
- K weights → `in_proj_b` (B = gate/control signal)
- V weights → `in_proj_x` (X = input content)

### Layer Plan
- 12-layer Pythia: keep {0, 3, 7, 11} as attention (4 anchors), convert 8 to Mamba
- 24-layer Qwen: keep {0, 4, 8, 12, 16, 23} as attention, convert 18 to Mamba (Step 2)

### Step 2 Results (Qwen2.5-0.5B, Colab T4, real mamba-ssm kernels)
- Model: Qwen2.5-0.5B (qwen2, 24 layers, 896 hidden, 14Q/2KV GQA, head_dim=64)
- Colab environment: Python 3.12, PyTorch 2.10.0+cu128, CUDA 12.8
- Layer plan: 18 RealMamba + 6 Attention (kept {0,4,8,12,16,23})
- Total params: 553,076,352 | Mamba+sink params: 92,094,464
- Cosine similarity: 0.354 (lower than Pythia — expected: Qwen2 sequential residual propagates Mamba changes further)
- Top-1 agreement: 0% (conversion changes all predictions — correct)
- No NaN, shape correct [1, 14, 151936] (10 tokens + 4 sinks), forward pass clean
- Generation: completes (gibberish — expected pre-distillation)
- **Real Mamba kernels confirmed executing on T4 GPU**

### Colab Environment Notes
- causal-conv1d and mamba-ssm must be compiled from source (~35-40 min total on T4)
- Do NOT `pip install torch` in setup script — Colab's pre-installed CUDA torch gets replaced by CPU-only PyPI torch, breaking compilation
- flash-attn fails to build (not needed for Steps 2-4)
- bitsandbytes and datasets install fine as pre-built wheels

### Step 1 Results
- Cosine similarity: 0.996 (high due to parallel residual + 4 kept attention layers)
- Top-1 token agreement: 0% (confirms conversion changes all predictions)
- Generation: gibberish (expected — distillation in Step 3 fixes this)
- No NaN, shapes correct, forward pass clean

## Running Step 1

```bash
cd "/run/media/deck/Katze2/Project Ouroboros"
source .venv/bin/activate
export HF_HOME="/run/media/deck/Katze2/Project Ouroboros/.cache/huggingface"
python scripts/convert_and_test.py --model EleutherAI/pythia-160m --device cpu
```

## Reference Documents
- `project ouroboros.docx` — Full 5-step spec with decision matrices and success criteria
- `ChatGPT responses.txt` — Scaffold code templates and v1.2 corrections (3625 lines of design notes)

## Key Design Principles
- FallbackMamba is deliberately simple (gated linear unit, no recurrence) — tests wiring, not SSM quality
- Real Mamba kernels (mamba-ssm, CUDA-only) used in Steps 2-4 on Colab
- Sink tokens (4 learnable) prepended to give attention layers initial tokens
- All code built here becomes reusable scaffolding for the full 9B Chimera run
