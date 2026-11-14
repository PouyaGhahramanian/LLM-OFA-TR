#!/usr/bin/env python
"""Prequential accuracy (top) and drift score (bottom) over the 2008-2019 stream, averaged over the six models.

Counterpart of LLM-OFA Figure 3, drawn as two panels on a shared time axis (no dual y-axis).
Output: figures/stream_accuracy_drift.png (+ .pdf) and results/stream_windows.csv (table view of the figure).
"""
import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ofa_tr.plotting import INK, SERIES, SERIES_LABEL, apply_style, plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = ["albert", "bert", "deberta", "distilbert", "electra", "roberta"]
CONFIGS = ["static", "ofa", "ofa-adapt"]


def windows(run_dir: Path, w: int):
    p = pd.read_csv(run_dir / "predictions.csv.gz")
    n = len(p) // w
    correct = (p["label"] == p["prediction"]).to_numpy()[: n * w].reshape(n, w).mean(1) * 100
    end_dates = pd.to_datetime(p["date"].to_numpy()[np.arange(1, n + 1) * w - 1])
    drift = None
    if (run_dir / "drift.csv.gz").exists():
        d = pd.read_csv(run_dir / "drift.csv.gz")
        d["win"] = (d["step"] - 1) // w
        d = d[d["win"] < n].groupby("win")["drift_score"].mean().reindex(range(n)).interpolate().bfill()
        drift = (d / d.mean()).to_numpy()  # relative to the model's own mean drift
    return correct, end_dates, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--smooth", type=int, default=3, help="centred moving average over windows (display only)")
    ap.add_argument("--settings", nargs="+", default=["pretrained", "finetuned"])
    args = ap.parse_args()
    apply_style()

    titles = {"pretrained": "(a) Önceden eğitilmiş modeller", "finetuned": "(b) İnce-ayarlı modeller"}
    fig, axes = plt.subplots(2, len(args.settings), figsize=(7.2, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    table = []
    for j, setting in enumerate(args.settings):
        acc_ax, drift_ax = axes[0, j], axes[1, j]
        curves, drifts, dates = {}, [], None
        for c in CONFIGS:
            per_model = []
            for m in MODELS:
                rd = REPO / "results" / "stream" / f"{setting}_{m}_{c}"
                if not (rd / "predictions.csv.gz").exists():
                    continue
                acc, dts, dr = windows(rd, args.window)
                per_model.append(acc)
                dates = dts
                if dr is not None:
                    drifts.append(dr)
            if per_model:
                curves[c] = np.mean(per_model, axis=0)
        smoothed = {}
        for c in CONFIGS:
            if c not in curves:
                continue
            smoothed[c] = pd.Series(curves[c]).rolling(args.smooth, center=True, min_periods=1).mean()
            acc_ax.plot(dates, smoothed[c], color=SERIES[c], label=SERIES_LABEL[c])
            table.append(pd.DataFrame({"setting": setting, "config": c, "window_end": dates,
                                       "mean_prequential_accuracy": curves[c]}))
        # direct end labels only where they do not collide (the legend carries identity otherwise)
        ends = {c: float(v.iloc[-1]) for c, v in smoothed.items()}
        lo, hi = acc_ax.get_ylim()
        gap = 0.05 * (hi - lo)
        for c, yv in ends.items():
            if all(abs(yv - o) > gap for k, o in ends.items() if k != c):
                acc_ax.annotate(SERIES_LABEL[c], (dates[-1], yv), xytext=(4, 0), textcoords="offset points",
                                va="center", fontsize=7.5, color=INK["secondary"])
        if drifts:
            dmean = pd.Series(np.mean(drifts, axis=0)).rolling(args.smooth, center=True, min_periods=1).mean()
            drift_ax.plot(dates, dmean, color=INK["secondary"], lw=1.4)
            table.append(pd.DataFrame({"setting": setting, "config": "drift(relative)", "window_end": dates,
                                       "mean_prequential_accuracy": np.mean(drifts, axis=0)}))
        acc_ax.set_title(titles[setting], loc="left")
        acc_ax.set_ylabel("Prequential doğruluk (%)" if j == 0 else "")
        drift_ax.set_yscale("log")
        if drifts:
            span = np.log10(float(np.max(dmean)) / max(float(np.min(dmean)), 1e-12))
            drift_ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,) if span > 1.5 else (1.0, 2.0, 5.0)))
        drift_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        drift_ax.yaxis.set_minor_locator(NullLocator())
        drift_ax.yaxis.set_minor_formatter(NullFormatter())
        drift_ax.set_ylabel("Kayma puanı\n(göreli, log)" if j == 0 else "")
        drift_ax.set_xlabel("Yıl")
        drift_ax.xaxis.set_major_locator(mdates.YearLocator(2))
        drift_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        acc_ax.margins(x=0.01)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(SERIES_LABEL[c]) for c in ["ofa-adapt", "ofa", "static"] if SERIES_LABEL[c] in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 0.97, 0.95))
    out = REPO / "figures" / "stream_accuracy_drift"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(f"{out}.png", dpi=300)
    fig.savefig(f"{out}.pdf")
    if table:
        pd.concat(table).to_csv(REPO / "results" / "stream_windows.csv", index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
