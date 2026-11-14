#!/usr/bin/env python
"""On-the-fly adaptation (OFA) stream evaluation on THD-Temporal (2008-2019 stream).

Configurations (same protocol as LLM-OFA, CIKM 2025):
  static     model only predicts, never updated
  ofa        test-then-train: each batch is predicted, then its labels are revealed and the model is updated
             with AdamW (lr 2e-5)
  ofa-adapt  same, but updated with Adaptimizer (lr 2e-5, window w=10, slow update rate rho=0.1, eps 1e-8).
             Predictions use the slow weights, updates the fast weights. Drift score = cosine distance between
             fast and slow weights; when it exceeds `--drift-threshold` x its running mean (after 10 steps) the
             slow weights are re-synchronised to the fast weights (as in the LLM-OFA runs, threshold 2).
Settings:
  pretrained the Hugging Face checkpoint; the classification head is calibrated with one pass over a random 1%
             of the stream (those items are then removed from the stream)
  finetuned  checkpoint produced by scripts/finetune.py on 1997-2007

Example:
  python scripts/stream.py --model bert --setting finetuned --config ofa-adapt
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.nn.functional import cross_entropy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ofa_tr import LABEL2ID, LABELS, MODELS, Adaptimizer, load_model, load_thd, prepare_texts, temporal_split  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_accuracy(correct: np.ndarray, w: int) -> np.ndarray:
    """Prequential accuracy over consecutive windows of w instances (last partial window dropped if tiny)."""
    n = len(correct) // w
    acc = correct[: n * w].reshape(n, w).mean(axis=1)
    rest = correct[n * w:]
    if len(rest) >= w // 2:
        acc = np.append(acc, rest.mean())
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--setting", required=True, choices=["pretrained", "finetuned"])
    ap.add_argument("--config", required=True, choices=["static", "ofa", "ofa-adapt"])
    ap.add_argument("--data", type=Path, default=None, help="path to thd_temporal.parquet")
    ap.add_argument("--models-dir", type=Path, default=REPO / "models")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "stream")
    ap.add_argument("--last-finetune-year", type=int, default=2007)
    ap.add_argument("--batch-size", type=int, default=None, help="default: per-model value from LLM-OFA")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--window-size", type=int, default=10, help="Adaptimizer squared-gradient window w")
    ap.add_argument("--slow-update-rate", type=float, default=0.1, help="Adaptimizer rho")
    ap.add_argument("--drift-metric", default="cosine", choices=["cosine", "l2", "mad"])
    ap.add_argument("--drift-threshold", type=float, default=2.0, help="<= 0 disables drift-triggered re-sync")
    ap.add_argument("--min-history", type=int, default=10)
    ap.add_argument("--calibration-frac", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=160)
    ap.add_argument("--eval-window", type=int, default=200, help="prequential accuracy window (instances)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N stream items (smoke test)")
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = args.batch_size or MODELS[args.model]["batch_size"]
    run_name = f"{args.setting}_{args.model}_{args.config}{args.tag}"
    out = args.out_dir / run_name
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- data
    df = load_thd(args.data)
    _, stream = temporal_split(df, args.last_finetune_year)
    if args.limit:
        stream = stream.iloc[: args.limit].reset_index(drop=True)

    # ---------------- model
    if args.setting == "finetuned":
        ckpt = args.models_dir / f"{args.model}_finetuned"
        if not (ckpt / "config.json").exists():
            sys.exit(f"missing fine-tuned checkpoint {ckpt}; run scripts/finetune.py --model {args.model} first")
        tokenizer, model = load_model(args.model, checkpoint=str(ckpt))
    else:
        tokenizer, model = load_model(args.model)
    commit = getattr(model.config, "_commit_hash", None)
    model.to(device)

    def encode(texts):
        enc = tokenizer(prepare_texts(texts, args.model), padding=True, truncation=True,
                        max_length=args.max_length, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    # ---------------- pretrained: calibrate the classification head on a random 1% of the stream
    n_calib = 0
    t_start = time.time()
    if args.setting == "pretrained":
        calib = stream.sample(frac=args.calibration_frac, random_state=args.seed)
        stream = stream.drop(index=calib.index).reset_index(drop=True)
        n_calib = len(calib)
        cal_opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        cbs = max(1, bs // 4)
        model.train()
        for i in range(0, n_calib, cbs):
            b = calib.iloc[i:i + cbs]
            y = torch.tensor([LABEL2ID[l] for l in b["label"]], device=device)
            loss = cross_entropy(model(**encode(b["text"].tolist())).logits, y)
            loss.backward()
            cal_opt.step()
            cal_opt.zero_grad(set_to_none=True)
        del cal_opt
        model.eval()

    # ---------------- online optimiser (created after calibration so slow weights start from it)
    opt = None
    if args.config == "ofa":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)  # betas (0.9, 0.999), eps 1e-8, wd 0.01
    elif args.config == "ofa-adapt":
        opt = Adaptimizer(model.parameters(), lr=args.lr, window_size=args.window_size,
                          slow_update_rate=args.slow_update_rate)

    # ---------------- stream: test-then-train
    preds, drift_rows, hist = [], [], []
    n_sync = 0
    model.eval()
    t0 = time.time()
    for i in range(0, len(stream), bs):
        b = stream.iloc[i:i + bs]
        enc = encode(b["text"].tolist())
        y = torch.tensor([LABEL2ID[l] for l in b["label"]], device=device)
        if args.config == "ofa-adapt":
            opt.apply_weights("slow")
        with torch.no_grad():
            preds.extend(model(**enc).logits.argmax(dim=1).tolist())
        if opt is not None:
            if args.config == "ofa-adapt":
                opt.restore_fast_weights()
            model.train()
            loss = cross_entropy(model(**enc).logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            model.eval()
            if args.config == "ofa-adapt":
                ds = opt.compute_drift_score(metric=args.drift_metric)
                hist.append(ds)
                synced = False
                if args.drift_threshold > 0 and len(hist) >= args.min_history \
                        and ds > args.drift_threshold * statistics.mean(hist):
                    opt.sync_slow_weights()
                    synced = True
                    n_sync += 1
                drift_rows.append({"step": i + len(b), "date": b["date"].iloc[-1], "drift_score": ds,
                                   "synced": synced})
        if (i // bs) % 200 == 0:
            done = i + len(b)
            acc_so_far = np.mean(np.array(preds) == stream["label"].map(LABEL2ID).values[:done])
            print(f"[{run_name}] {done}/{len(stream)} acc={acc_so_far:.4f} "
                  f"{(time.time() - t0) / max(done, 1) * 1000:.1f} ms/item", flush=True)
    runtime = time.time() - t0

    # ---------------- metrics
    y_true = stream["label"].map(LABEL2ID).values
    y_pred = np.array(preds)
    correct = (y_true == y_pred).astype(float)
    win = window_accuracy(correct, args.eval_window)
    yearly = pd.DataFrame({"year": stream["year"], "c": correct}).groupby("year")["c"].mean()
    metrics = {
        "run": run_name, "model": args.model, "hf_id": MODELS[args.model]["hf_id"], "hf_commit": commit,
        "setting": args.setting, "config": args.config, "n_stream": int(len(stream)), "n_calibration": n_calib,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(LABELS))),
                                   zero_division=0)),
        "eval_window": args.eval_window, "n_windows": int(len(win)),
        "min_window_accuracy": float(win.min()), "mean_window_accuracy": float(win.mean()),
        "yearly_accuracy": {int(k): float(v) for k, v in yearly.items()},
        "drift_syncs": n_sync, "runtime_sec": round(runtime, 1),
        "runtime_total_sec": round(time.time() - t_start, 1), "batch_size": bs,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "python": platform.python_version(),
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
    }
    pd.DataFrame({"position": np.arange(len(stream)), "id": stream["id"], "date": stream["date"],
                  "label": stream["label"], "prediction": [LABELS[p] for p in y_pred]}
                 ).to_csv(out / "predictions.csv.gz", index=False)
    if drift_rows:
        pd.DataFrame(drift_rows).to_csv(out / "drift.csv.gz", index=False)
    np.savetxt(out / "window_accuracy.txt", win, fmt="%.6f")
    (out / "report.txt").write_text(classification_report(
        y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS, digits=4, zero_division=0),
        encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("run", "accuracy", "macro_f1", "min_window_accuracy",
                                               "drift_syncs", "runtime_sec")}, indent=1))


if __name__ == "__main__":
    main()
