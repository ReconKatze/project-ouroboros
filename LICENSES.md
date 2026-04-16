# Third-Party Licenses

This file documents all third-party software that Project Amore / Project Chimera
depends on, references, or includes in this repository.

---

## hermes-agent (Nous Research)

**Source:** https://github.com/NousResearch/hermes-agent  
**Version used:** 0.9.0 (downloaded April 2026)  
**License:** MIT License  
**Copyright:** Copyright (c) 2025 Nous Research

`chimera/deployment/context_engine.py` and `chimera/deployment/think_bridge.py`
are designed as hermes-agent plugins.  They inherit from hermes-agent's
`ContextEngine` ABC and are dropped into `plugins/context_engine/chimera/`
at deployment time.

```
MIT License

Copyright (c) 2025 Nous Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Mamba / mamba-ssm (Tri Dao, Albert Gu)

**Source:** https://github.com/state-spaces/mamba  
**License:** Apache License 2.0  
**Copyright:** Copyright (c) 2023 Tri Dao, Albert Gu

Used for `RealMambaBlock` / `Mamba3Block` SSM kernels in Steps 2–4 (Colab A100).
Not bundled in this repository; installed at runtime via:
`MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git`

---

## Qwen2.5 Models (Alibaba Cloud)

**Source:** https://huggingface.co/Qwen  
**License:** Qwen License (permissive for research; commercial use above 100M MAU
requires separate agreement — see https://huggingface.co/Qwen/Qwen2.5-Coder-7B)  
**Copyright:** Copyright (c) 2024 Alibaba Cloud

`Qwen2.5-Coder-7B` — teacher model for knowledge distillation (Step 4).  
`Qwen2.5-Coder-1.5B` — student model backbone (Step 4).  
Neither model is bundled; downloaded at runtime from HuggingFace Hub.

---

## PyTorch (Meta)

**Source:** https://github.com/pytorch/pytorch  
**License:** BSD 3-Clause ("New BSD") License  
**Copyright:** Copyright (c) 2016-present, Facebook, Inc.  
Not bundled; installed as a dependency (`torch>=2.10`).

---

## HuggingFace Transformers

**Source:** https://github.com/huggingface/transformers  
**License:** Apache License 2.0  
**Copyright:** Copyright 2018-present The HuggingFace Inc. team  
Not bundled; installed as a dependency (`transformers>=5.4`).

---

## Project Amore / Project Chimera (this repository)

All original code and documentation in this repository is the work of the author
(ReconKatze) and is published for research and educational purposes.  No
separate open-source license is attached at this time.  Contact:
projectamore26@gmail.com.
