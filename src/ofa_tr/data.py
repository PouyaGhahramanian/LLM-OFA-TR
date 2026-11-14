"""THD-Temporal data utilities.

THD-Temporal is built from the Turkish News Corpus (Türkçe Haber Derlemi) collected by Ömer Gözüaçık
(https://github.com/ogozuacik/turkce-haber-derlemi, MIT License). See scripts/prepare_thd_temporal.py.
"""
from pathlib import Path

import pandas as pd

LABELS = ["Diğer", "Dünya", "Ekonomi", "Gündem", "Kültür-Sanat", "Magazin", "Siyaset", "Spor",
          "Teknoloji-Bilim", "Yaşam"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "data" / "thd_temporal.parquet"
LAST_FINETUNE_YEAR = 2007  # 1997-2007 -> initial fine-tuning, 2008-2019 -> stream
COLUMNS = ["id", "source_row", "date", "year", "title", "summary", "label"]


def turkish_lower(text: str) -> str:
    """Lower-case Turkish text correctly (I -> ı, İ -> i) before feeding uncased tokenizers."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def make_text(title: str, summary: str) -> str:
    """Model input: title + summary (summary is skipped when it is empty or repeats the title)."""
    title, summary = (title or "").strip(), (summary or "").strip()
    if not summary or summary == title:
        return title
    sep = " " if title[-1:] in ".!?…:;\"'" else ". "
    return f"{title}{sep}{summary}"


def load_thd(path=None, columns=COLUMNS) -> pd.DataFrame:
    """Load THD-Temporal in chronological order and add the `text` model input."""
    df = pd.read_parquet(path or DEFAULT_PATH, columns=columns)
    df = df.sort_values(["date", "source_row"], kind="stable").reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df["text"] = [make_text(t, s) for t, s in zip(df["title"], df["summary"])]
    return df


def temporal_split(df: pd.DataFrame, last_finetune_year: int = LAST_FINETUNE_YEAR):
    """Chronological split into the initial fine-tuning period and the evaluation stream."""
    finetune = df[df["year"] <= last_finetune_year].reset_index(drop=True)
    stream = df[df["year"] > last_finetune_year].reset_index(drop=True)
    return finetune, stream
