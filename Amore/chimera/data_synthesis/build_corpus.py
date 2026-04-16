"""build_corpus.py — Assemble all generated JSONL files into training splits.

Takes the output files from all generators and produces:

  train_sft.jsonl   — SFT corpus: all {"text": ...} records, shuffled
  train_dpo.jsonl   — DPO corpus: all {"prompt", "chosen", "rejected"} records
  stats.json        — record counts per source category

The corpus is stratified: constitutional and error pairs are upsampled
relative to raw CoT to balance representation during training.

Usage::

    from chimera.data_synthesis.build_corpus import build_corpus
    build_corpus(
        output_dir="data/synthesis",
        sft_upsample={"constitutional:anchor": 3, "constitutional:adversarial": 3},
    )
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union


# Default upsampling multipliers by source prefix
DEFAULT_UPSAMPLE: Dict[str, int] = {
    "synthesis:cot":                        1,
    "synthesis:constitutional:anchor":      4,   # identity-shaping, most critical
    "synthesis:constitutional:adversarial": 4,   # constitutional hardening
    "synthesis:constitutional:cultural":    2,   # cultural attractor
    "synthesis:preferences":               2,    # alignment
    "synthesis:error_pairs":               2,    # correctness
}


def _read_jsonl(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_corpus(
    output_dir: Union[str, Path],
    sft_upsample: Optional[Dict[str, int]] = None,
    seed: int = 42,
    val_fraction: float = 0.05,
) -> Dict[str, int]:
    """Assemble generated files into train/val splits.

    Parameters
    ----------
    output_dir :
        Directory containing the individual JSONL files from each generator.
        Output files are written to the same directory.
    sft_upsample :
        Per-source upsampling multipliers. Keys match the ``source`` field in
        JSONL records. Defaults to ``DEFAULT_UPSAMPLE``.
    seed :
        Random seed for shuffle and split.
    val_fraction :
        Fraction of SFT records held out for validation.

    Returns
    -------
    Dict mapping split name to record count.
    """
    output_dir = Path(output_dir)
    upsample = sft_upsample or DEFAULT_UPSAMPLE

    # --- Collect SFT records ({"text": ...}) ---
    sft_files = [
        output_dir / "cot_traces.jsonl",
        output_dir / "anchor_responses.jsonl",
        output_dir / "adversarial_responses.jsonl",
        output_dir / "cultural_scenarios.jsonl",
        output_dir / "preferences_sft.jsonl",
        output_dir / "error_pairs.jsonl",
    ]

    sft_all: List[Dict] = []
    source_counts: Counter = Counter()

    for path in sft_files:
        records = _read_jsonl(path)
        for record in records:
            if "text" not in record:
                continue
            # Only keep "chosen" label records from SFT preference file
            if record.get("label") == "rejected":
                continue
            source = record.get("source", "unknown")
            multiplier = 1
            for prefix, mult in upsample.items():
                if source.startswith(prefix):
                    multiplier = mult
                    break
            for _ in range(multiplier):
                sft_all.append(record)
            source_counts[source] += 1

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(sft_all)

    val_n = max(1, int(len(sft_all) * val_fraction))
    val_sft = sft_all[:val_n]
    train_sft = sft_all[val_n:]

    _write_jsonl(output_dir / "train_sft.jsonl", train_sft)
    _write_jsonl(output_dir / "val_sft.jsonl", val_sft)

    # --- Collect DPO records ({"prompt", "chosen", "rejected"}) ---
    dpo_records = _read_jsonl(output_dir / "preferences_dpo.jsonl")
    rng.shuffle(dpo_records)

    val_dpo_n = max(1, int(len(dpo_records) * val_fraction))
    val_dpo = dpo_records[:val_dpo_n]
    train_dpo = dpo_records[val_dpo_n:]

    _write_jsonl(output_dir / "train_dpo.jsonl", train_dpo)
    _write_jsonl(output_dir / "val_dpo.jsonl", val_dpo)

    # --- Stats ---
    stats = {
        "train_sft": len(train_sft),
        "val_sft": len(val_sft),
        "train_dpo": len(train_dpo),
        "val_dpo": len(val_dpo),
        "total_sft_before_upsample": sum(source_counts.values()),
        "total_sft_after_upsample": len(sft_all),
        "by_source": dict(source_counts),
    }
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\n=== Corpus built ===")
    print(f"  SFT train: {stats['train_sft']:,}  val: {stats['val_sft']:,}")
    print(f"  DPO train: {stats['train_dpo']:,}  val: {stats['val_dpo']:,}")
    print(f"  Sources:")
    for src, count in sorted(source_counts.items()):
        mult = 1
        for prefix, m in upsample.items():
            if src.startswith(prefix):
                mult = m
                break
        print(f"    {src}: {count} raw × {mult} = {count * mult}")

    return stats
