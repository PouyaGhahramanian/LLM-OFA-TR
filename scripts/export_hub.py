#!/usr/bin/env python
"""Build upload-ready folders for the Hugging Face Hub and Kaggle from data/*.parquet (no upload is done here).

  python scripts/export_hub.py --hf-id <user>/thd-temporal --kaggle-id <user>/thd-temporal \
      --github-url https://github.com/<user>/LLM-OFA-TR

Creates hub_export/hf/ (dataset card README.md with YAML configs, data/{finetune,stream}.parquet,
full/thd_temporal_full.parquet, LICENSE, stats.json) and hub_export/kaggle/ (parquet + csv.gz, README.md,
LICENSE, dataset-metadata.json). Upload afterwards with `hf upload` / `kaggle datasets create`.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def label_table(stats):
    labels = sorted(stats["label_counts"], key=lambda k: -stats["label_counts"][k])
    ft, st = stats["label_counts_by_split"]["finetune"], stats["label_counts_by_split"]["stream"]
    rows = ["| label | finetune (1997–2007) | stream (2008–2019) | total |", "|---|---:|---:|---:|"]
    for lab in labels:
        rows.append(f"| {lab} | {ft.get(lab, 0):,} | {st.get(lab, 0):,} | {stats['label_counts'][lab]:,} |")
    rows.append(f"| **total** | **{sum(ft.values()):,}** | **{sum(st.values()):,}** | **{stats['n_benchmark']:,}** |")
    return "\n".join(rows)


def render_card(stats, args):
    card = (REPO / "hub" / "DATASET_CARD.md").read_text(encoding="utf-8")
    values = {
        "N_BENCH": stats["n_benchmark"], "N_DATE_FIX": stats["dates_repaired_day_month_swap"],
        "N_2020": stats["excluded"].get("out_of_range_year", 0), "N_DUP": stats["excluded"].get("duplicate", 0),
        "N_FT": stats["split_sizes"]["finetune"], "N_ST": stats["split_sizes"]["stream"],
        "Y_MIN": stats["yearly"]["min"], "Y_MIN_YEAR": stats["yearly"]["min_year"],
        "Y_MAX": stats["yearly"]["max"], "Y_MAX_YEAR": stats["yearly"]["max_year"],
        "Y_MEAN": stats["yearly"]["mean"], "COL_SHARE": stats["columns_share"],
        "LABEL_TABLE": label_table(stats), "HF_ID": args.hf_id, "KAGGLE_ID": args.kaggle_id,
        "GITHUB_URL": args.github_url,
    }
    # fill {KEY} / {KEY:fmt} placeholders without touching other braces (YAML, BibTeX)
    import re

    def sub(m):
        key, fmt = m.group(1), m.group(2) or ""
        return format(values[key], fmt[1:]) if fmt else str(values[key])

    return re.sub(r"\{([A-Z_0-9]+)(:[^}]*)?\}", lambda m: sub(m) if m.group(1) in values else m.group(0), card)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-id", default="<hf-user>/thd-temporal")
    ap.add_argument("--kaggle-id", default="<kaggle-user>/thd-temporal")
    ap.add_argument("--github-url", default="https://github.com/<user>/LLM-OFA-TR")
    ap.add_argument("--out", type=Path, default=REPO / "hub_export")
    args = ap.parse_args()

    stats = json.loads((REPO / "data" / "stats.json").read_text(encoding="utf-8"))
    bench = pd.read_parquet(REPO / "data" / "thd_temporal.parquet")
    full = pd.read_parquet(REPO / "data" / "thd_temporal_full.parquet")
    card = render_card(stats, args)
    lic = REPO / "data" / "LICENSE-THD.txt"

    hf = args.out / "hf"
    if hf.exists():
        shutil.rmtree(hf)
    (hf / "data").mkdir(parents=True)
    (hf / "full").mkdir()
    for split in ("finetune", "stream"):
        bench[bench["split"] == split].to_parquet(hf / "data" / f"{split}.parquet", index=False, compression="zstd")
    full.to_parquet(hf / "full" / "thd_temporal_full.parquet", index=False, compression="zstd")
    (hf / "README.md").write_text(card, encoding="utf-8")
    shutil.copy(lic, hf / "LICENSE")
    shutil.copy(REPO / "data" / "stats.json", hf / "stats.json")

    kg = args.out / "kaggle"
    if kg.exists():
        shutil.rmtree(kg)
    kg.mkdir(parents=True)
    bench.to_parquet(kg / "thd_temporal.parquet", index=False, compression="zstd")
    full.to_parquet(kg / "thd_temporal_full.parquet", index=False, compression="zstd")
    bench.to_csv(kg / "thd_temporal.csv.gz", index=False)
    # Kaggle shows README.md-like text from the metadata description; strip the YAML front matter
    body = card.split("---", 2)[2].strip() if card.startswith("---") else card
    (kg / "README.md").write_text(body, encoding="utf-8")
    shutil.copy(lic, kg / "LICENSE")
    meta = {
        "title": "THD-Temporal: Turkish News Stream 1997-2019",
        "subtitle": "78K Milliyet news articles over 23 years for concept-drift and online learning",
        "id": args.kaggle_id,
        "licenses": [{"name": "MIT"}],
        "keywords": ["nlp", "news", "text", "classification"],
        "description": body,
    }
    (kg / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in sorted(args.out.rglob("*")):
        if p.is_file():
            print(f"{p.relative_to(args.out)}  {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
