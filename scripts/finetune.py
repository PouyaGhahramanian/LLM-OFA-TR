#!/usr/bin/env python
"""Fine-tune a Turkish encoder on the first period of THD-Temporal (1997-2007), as in LLM-OFA.

Protocol (mirrors LLM-OFA/src/finetune.py): stratified 90/10 train/validation split of the fine-tuning period,
Hugging Face Trainer, 3 epochs, batch size 32, default learning rate 5e-5 with linear decay, best epoch selected
by validation loss. Inputs are title + summary, truncated to 160 sub-word tokens.

Example:
  python scripts/finetune.py --model bert
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ofa_tr import LABEL2ID, MODELS, load_model, load_thd, prepare_texts, temporal_split  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--models-dir", type=Path, default=REPO / "models")
    ap.add_argument("--last-finetune-year", type=int, default=2007)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=160)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="subsample N items (smoke test)")
    args = ap.parse_args()
    set_seed(args.seed)

    out = args.models_dir / f"{args.model}_finetuned"
    df = load_thd(args.data)
    ft, _ = temporal_split(df, args.last_finetune_year)
    if args.limit:
        ft = ft.sample(args.limit, random_state=args.seed)
        out = args.models_dir / f"{args.model}_finetuned_smoke"
    x_tr, x_ev, y_tr, y_ev = train_test_split(ft["text"].tolist(), ft["label"].tolist(), test_size=args.eval_frac,
                                              random_state=args.seed, stratify=ft["label"])
    tokenizer, model = load_model(args.model)

    def build(texts, labels):
        enc = tokenizer(prepare_texts(texts, args.model), truncation=True, max_length=args.max_length)
        return EncodedDataset(enc, [LABEL2ID[l] for l in labels])

    train_ds, eval_ds = build(x_tr, y_tr), build(x_ev, y_ev)

    def compute_metrics(p):
        pred = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, pred),
                "macro_f1": f1_score(p.label_ids, pred, average="macro")}

    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
        logging_steps=100, report_to="none", seed=args.seed, dataloader_num_workers=2)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    train_result = trainer.train()
    final_eval = trainer.evaluate()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    for ck in out.glob("checkpoint-*"):
        shutil.rmtree(ck)
    summary = {"model": args.model, "hf_id": MODELS[args.model]["hf_id"], "n_train": len(train_ds),
               "n_validation": len(eval_ds), "train_loss": train_result.training_loss,
               "train_runtime_sec": train_result.metrics.get("train_runtime"),
               "validation": {k: float(v) for k, v in final_eval.items() if isinstance(v, (int, float))},
               "log_history": trainer.state.log_history, "args": {k: str(v) for k, v in vars(args).items()}}
    (out / "finetune_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("model", "n_train", "train_loss", "validation")}, indent=1))


if __name__ == "__main__":
    main()
