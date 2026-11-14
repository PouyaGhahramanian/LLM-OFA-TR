#!/usr/bin/env python
"""Build THD-Temporal from the Turkish News Corpus (Türkçe Haber Derlemi) of Ömer Gözüaçık.

Original corpus (MIT License, Copyright (c) 2020 Ömer Gözüaçık):
    https://github.com/ogozuacik/turkce-haber-derlemi
    * filtrelenmis_temizlenmis_derlem.csv.gz : 81,407 Milliyet.com articles (1997-2019) selected and labelled
      by the original author into 12 categories. Text is lower-cased, numbers/punctuation removed.
    * milliyet_derlem.csv.gz                 : the raw scrape with the original (cased, punctuated) text.

THD-Temporal keeps the original author's article selection and labels and
  1. restores the original text (title / summary / body) by joining both files on `link`,
  2. repairs publication dates (the filtered file has day/month swapped for dates with day <= 12),
  3. adds a 10-class `label` (Ege, Pazar and Cumartesi -- regional/weekend supplements -- merged into "Diğer"),
  4. flags invalid dates (1899), articles dated only by the June/July 2020 crawl (mostly Skorer sports articles
     published ~2009-2019, exact dates unknown) and exact duplicate articles,
  5. defines a chronological benchmark split: `finetune` (1997-2007) and `stream` (2008-2019).

Usage:
    python scripts/prepare_thd_temporal.py --orig-dir /path/to/originals --out-dir data
(the two original files are downloaded from GitHub LFS into --orig-dir if they are missing)
"""
import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

GITHUB_LFS = "https://media.githubusercontent.com/media/ogozuacik/turkce-haber-derlemi/master/derlemler/"
FILTERED = "filtrelenmis_temizlenmis_derlem.csv.gz"
RAW = "milliyet_derlem.csv.gz"
MD5 = {FILTERED: "acf2be55c51d6b268c9d935d9f58de61", RAW: "d32560c1a935e8662f6c3c621568e28b"}
BASE_URL = "https://www.milliyet.com.tr"

# original 12-class `category` -> THD-Temporal 10-class `label`
LABEL_MAP = {
    "Gündem": "Gündem", "Siyaset": "Siyaset", "Ekonomi": "Ekonomi", "Spor": "Spor", "Dünya": "Dünya",
    "Magazin": "Magazin", "Teknoloji_Bilim": "Teknoloji-Bilim", "Yaşam": "Yaşam", "Kültür_Sanat": "Kültür-Sanat",
    "Ege": "Diğer", "Pazar": "Diğer", "Cumartesi": "Diğer",
}
FIRST_YEAR, LAST_YEAR = 1997, 2019
NON_AUTHOR_SLUGS = {"galeri", "oyun-dunyasi", "teknoloji-haberleri"}  # 3-level paths that are not columnists
ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(orig_dir: Path, name: str) -> Path:
    path = orig_dir / name
    if not path.exists():
        print(f"downloading {name} ...")
        orig_dir.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(GITHUB_LFS + name, headers={"User-Agent": "thd-temporal-builder"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
            f.write(r.read())
    digest = md5(path)
    if digest != MD5[name]:
        print(f"WARNING: md5 of {name} is {digest}, expected {MD5[name]} (upstream file changed?)")
    return path


def clean_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)
    s = ZERO_WIDTH.sub("", s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-dir", type=Path, required=True, help="folder with (or for) the original csv.gz files")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--last-finetune-year", type=int, default=2007)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    flt = pd.read_csv(fetch(args.orig_dir, FILTERED))
    raw = pd.read_csv(fetch(args.orig_dir, RAW))
    print(f"filtered: {flt.shape}, raw: {raw.shape}")

    n_raw_rows = len(raw)
    flt = flt.reset_index().rename(columns={"index": "source_row"})
    # a few links occur more than once (25 rows in the filtered file, 49 in the raw scrape): match the i-th
    # occurrence in the filtered file with the i-th occurrence in the raw scrape
    flt["occ"] = flt.groupby("link").cumcount()
    raw = raw.assign(occ=raw.groupby("link").cumcount())
    df = flt.merge(raw[["link", "occ", "news_title", "summary", "news", "date"]], on=["link", "occ"], how="left",
                   suffixes=("_flt", ""), validate="one_to_one")
    assert df["news_title"].notna().all(), "some filtered articles have no raw counterpart"

    # --- dates: raw dates are dd.mm.yyyy; the filtered file parsed them month-first when day <= 12
    d_raw = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    d_flt = pd.to_datetime(df["date_flt"], errors="coerce")
    swapped = (d_raw.notna() & d_flt.notna() & (d_raw != d_flt) & (d_raw.dt.year == d_flt.dt.year)
               & (d_raw.dt.day == d_flt.dt.month) & (d_raw.dt.month == d_flt.dt.day))
    other_mismatch = d_raw.notna() & d_flt.notna() & (d_raw != d_flt) & ~swapped
    date = d_raw.fillna(d_flt)

    # opinion columns: /yazarlar/<author>/<slug> and /<section>/<author>/<slug> (e.g. Cadde, Skorer, Pembenar)
    parts = df["link"].str.extract(r"^/([^/]+)/([^/]+)/[^/]+$")
    is_col = (parts[0].eq("yazarlar") | (parts[0].notna() & ~parts[1].isin(NON_AUTHOR_SLUGS))).fillna(False)

    out = pd.DataFrame({
        "id": [f"thd-{i:06d}" for i in df["source_row"]],
        "source_row": df["source_row"],
        "date": date.dt.strftime("%Y-%m-%d"),
        "year": date.dt.year.astype("Int64"),
        "title": df["news_title"].map(clean_text),
        "summary": df["summary"].map(clean_text),
        "news": df["news"].map(clean_text),
        "label": df["category"].map(LABEL_MAP),
        "category": df["category"],
        "category_original": df["category_backup"],
        "link": df["link"],
        "url": BASE_URL + df["link"],
        "is_column": is_col.astype(bool),
        "author": parts[1].where(is_col),
        "date_repaired": swapped,
    })
    assert out["label"].notna().all(), "unmapped category"

    # --- benchmark eligibility
    reason = pd.Series("", index=out.index, dtype=object)
    reason[out["year"].isna() | (out["year"] < FIRST_YEAR)] = "invalid_date"
    reason[(reason == "") & (out["year"] > LAST_YEAR)] = "out_of_range_year"
    order = out.assign(_r=reason).sort_values(["date", "source_row"]).index
    eligible = out.loc[order][reason.loc[order] == ""]
    dup = eligible.duplicated(["title", "summary", "news"], keep="first") | eligible.duplicated("link", keep="first")
    reason[dup[dup].index] = "duplicate"
    out["exclusion"] = reason.replace("", None)
    out["in_benchmark"] = reason == ""
    out["split"] = None
    bench_mask = out["in_benchmark"]
    out.loc[bench_mask, "split"] = (out.loc[bench_mask, "year"] <= args.last_finetune_year).map(
        {True: "finetune", False: "stream"})

    out = out.sort_values(["date", "source_row"], na_position="first").reset_index(drop=True)
    bench = out[out["in_benchmark"]].drop(columns=["in_benchmark", "exclusion"]).reset_index(drop=True)

    out.to_parquet(args.out_dir / "thd_temporal_full.parquet", index=False)
    bench.to_parquet(args.out_dir / "thd_temporal.parquet", index=False)

    # --- statistics
    yearly = bench.groupby("year").size()
    stats = {
        "source": "https://github.com/ogozuacik/turkce-haber-derlemi (MIT, (c) 2020 Ömer Gözüaçık)",
        "n_filtered_original": int(len(flt)),
        "n_raw_original": int(n_raw_rows),
        "n_raw_unique_links": int(raw["link"].nunique()),
        "dates_repaired_day_month_swap": int(swapped.sum()),
        "dates_other_mismatch": int(other_mismatch.sum()),
        "excluded": out["exclusion"].value_counts().to_dict(),
        "n_benchmark": int(len(bench)),
        "split_sizes": bench["split"].value_counts().to_dict(),
        "label_counts": bench["label"].value_counts().to_dict(),
        "label_counts_by_split": {s: g["label"].value_counts().to_dict() for s, g in bench.groupby("split")},
        "yearly": {"min": int(yearly.min()), "min_year": int(yearly.idxmin()), "max": int(yearly.max()),
                   "max_year": int(yearly.idxmax()), "mean": round(float(yearly.mean()), 1),
                   "per_year": {int(k): int(v) for k, v in yearly.items()}},
        "columns_share": round(float(bench["is_column"].mean()), 4),
        "summary_equals_title": int((bench["summary"] == bench["title"]).sum()),
        "label_map": LABEL_MAP,
        "last_finetune_year": args.last_finetune_year,
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("label_map",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
