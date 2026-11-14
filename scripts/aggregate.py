#!/usr/bin/env python
"""Aggregate stream runs into result tables (CSV / Markdown / JSON).

Outputs (results/):
  summary.csv            one row per run
  table_main.md          Turkish counterpart of LLM-OFA Table 2 (accuracy | macro-F1, gains over Static)
  table_en_vs_tr.md      average gains on 1M-News (English, CIKM 2025 Table 2) vs THD-Temporal (Turkish)
  summary.json           averages used in the report
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
MODELS = ["albert", "bert", "deberta", "distilbert", "electra", "roberta"]
NAMES = {"albert": "ALBERT", "bert": "BERT", "deberta": "DeBERTa", "distilbert": "DistilBERT",
         "electra": "ELECTRA", "roberta": "RoBERTa"}
CONFIGS = ["static", "ofa", "ofa-adapt"]
SETTINGS = ["pretrained", "finetuned"]

# LLM-OFA (CIKM 2025) Table 2, 1M-News, accuracy / macro-F1 (%), order: static, ofa, ofa-adapt
EN = {
    "pretrained": {
        "albert": ((71.86, 83.43, 85.57), (68.15, 82.33, 84.77)),
        "bert": ((72.54, 86.79, 87.06), (63.46, 86.02, 86.26)),
        "deberta": ((78.47, 86.67, 87.00), (75.60, 85.98, 86.23)),
        "distilbert": ((65.29, 85.65, 85.87), (47.21, 84.76, 84.93)),
        "electra": ((64.56, 85.42, 86.41), (46.01, 84.51, 85.62)),
        "roberta": ((76.82, 87.14, 87.71), (73.00, 86.43, 87.03)),
    },
    "finetuned": {
        "albert": ((89.09, 89.67, 90.14), (88.24, 89.15, 89.65)),
        "bert": ((93.40, 94.02, 94.21), (92.85, 93.75, 93.92)),
        "deberta": ((89.72, 89.80, 90.07), (88.82, 89.19, 89.41)),
        "distilbert": ((92.32, 93.42, 93.60), (92.06, 93.10, 93.28)),
        "electra": ((91.22, 91.87, 92.18), (90.38, 91.48, 91.78)),
        "roberta": ((91.57, 92.46, 92.76), (90.90, 92.13, 92.43)),
    },
}


def load_runs():
    rows = []
    for f in sorted((RES / "stream").glob("*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.append({k: m[k] for k in ("run", "model", "setting", "config", "hf_id", "n_stream", "accuracy",
                                       "macro_f1", "min_window_accuracy", "mean_window_accuracy", "n_windows",
                                       "drift_syncs", "runtime_sec", "batch_size")})
    df = pd.DataFrame(rows)
    for c in ("accuracy", "macro_f1", "min_window_accuracy", "mean_window_accuracy"):
        df[c] = df[c] * 100
    return df


def fmt(v, best=False, gain=None):
    s = f"{v:.2f}"
    if gain is not None:
        s += f" ({gain:+.2f})"
    return f"**{s}**" if best else s


def main():
    df = load_runs()
    df.sort_values(["setting", "model", "config"]).to_csv(RES / "summary.csv", index=False)
    piv = {met: df.pivot_table(index=["setting", "model"], columns="config", values=met)
           for met in ("accuracy", "macro_f1", "min_window_accuracy")}

    # one table per metric; columns: pretrained (static, ofa, ofa-adapt) | fine-tuned (static, ofa, ofa-adapt)
    out = []
    for met, title in (("accuracy", "Doğruluk (accuracy, %)"), ("macro_f1", "Makro-F1 (%)")):
        out.append(f"\n### {title}\n")
        out.append("| LLM | Static | OFA | OFA-Adapt | Static | OFA | OFA-Adapt |")
        out.append("|---|---|---|---|---|---|---|")
        out.append("| | *önceden eğitilmiş* | | | *ince-ayarlı* | | |")
        for m in MODELS:
            cells = []
            for s in SETTINGS:
                try:
                    vals = [piv[met].loc[(s, m), c] for c in CONFIGS]
                except KeyError:
                    cells += ["–"] * 3
                    continue
                best = int(np.nanargmax(vals))
                for i, c in enumerate(CONFIGS):
                    gain = vals[i] - vals[0] if c == "ofa-adapt" else None
                    cells.append(fmt(vals[i], best == i, gain))
            out.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    (RES / "table_main.md").write_text("\n".join(out[1:]) + "\n", encoding="utf-8")

    # averages
    summ = {}
    for s in SETTINGS:
        d = {}
        for met in ("accuracy", "macro_f1", "min_window_accuracy"):
            p = piv[met].loc[s] if s in piv[met].index.get_level_values(0) else None
            if p is None or not set(CONFIGS) <= set(p.columns):
                continue
            p = p.dropna()
            d[met] = {
                "n_models": int(len(p)),
                "mean": {c: float(p[c].mean()) for c in CONFIGS},
                "gain_ofa_vs_static": float((p["ofa"] - p["static"]).mean()),
                "gain_adapt_vs_static": float((p["ofa-adapt"] - p["static"]).mean()),
                "gain_adapt_vs_ofa": float((p["ofa-adapt"] - p["ofa"]).mean()),
                "adapt_beats_ofa": int((p["ofa-adapt"] > p["ofa"]).sum()),
                "adapt_beats_static": int((p["ofa-adapt"] > p["static"]).sum()),
                "max_gain_adapt_vs_static": {"model": str(p["ofa-adapt"].sub(p["static"]).idxmax()),
                                             "value": float(p["ofa-adapt"].sub(p["static"]).max())},
                "max_gain_adapt_vs_ofa": {"model": str(p["ofa-adapt"].sub(p["ofa"]).idxmax()),
                                          "value": float(p["ofa-adapt"].sub(p["ofa"]).max())},
            }
        summ[s] = d
    # English averages from the paper
    en = {}
    for s in SETTINGS:
        acc = np.array([EN[s][m][0] for m in MODELS])
        f1 = np.array([EN[s][m][1] for m in MODELS])
        en[s] = {"accuracy": {"mean": dict(zip(CONFIGS, acc.mean(0))), "gain_ofa_vs_static": float((acc[:, 1] - acc[:, 0]).mean()),
                              "gain_adapt_vs_static": float((acc[:, 2] - acc[:, 0]).mean()),
                              "gain_adapt_vs_ofa": float((acc[:, 2] - acc[:, 1]).mean())},
                 "macro_f1": {"mean": dict(zip(CONFIGS, f1.mean(0))), "gain_ofa_vs_static": float((f1[:, 1] - f1[:, 0]).mean()),
                              "gain_adapt_vs_static": float((f1[:, 2] - f1[:, 0]).mean()),
                              "gain_adapt_vs_ofa": float((f1[:, 2] - f1[:, 1]).mean())}}
    rows = ["| Ayar | Metrik | Dil | Static | OFA | OFA-Adapt | OFA − Static | OFA-Adapt − Static | OFA-Adapt − OFA |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s, sn in (("pretrained", "Önceden eğitilmiş"), ("finetuned", "İnce-ayarlı")):
        for met, mn in (("accuracy", "Doğruluk"), ("macro_f1", "Makro-F1")):
            for lang, src in (("İngilizce (1M-News)", en), ("Türkçe (THD-Temporal)", summ)):
                d = src.get(s, {}).get(met)
                if not d:
                    continue
                mu = d["mean"]
                rows.append(f"| {sn} | {mn} | {lang} | {mu['static']:.2f} | {mu['ofa']:.2f} | {mu['ofa-adapt']:.2f} | "
                            f"{d['gain_ofa_vs_static']:+.2f} | {d['gain_adapt_vs_static']:+.2f} | {d['gain_adapt_vs_ofa']:+.2f} |")
    (RES / "table_en_vs_tr.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (RES / "summary.json").write_text(json.dumps({"turkish": summ, "english_cikm2025": en}, indent=2,
                                                 ensure_ascii=False), encoding="utf-8")
    print((RES / "table_main.md").read_text(encoding="utf-8"))
    print((RES / "table_en_vs_tr.md").read_text(encoding="utf-8"))
    print(json.dumps(summ, indent=1)[:3000])


if __name__ == "__main__":
    main()
