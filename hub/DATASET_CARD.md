---
language:
- tr
license: mit
pretty_name: "THD-Temporal: Turkish News Stream 1997-2019"
size_categories:
- 10K<n<100K
task_categories:
- text-classification
task_ids:
- topic-classification
tags:
- turkish
- news
- concept-drift
- temporal
- online-learning
- stream-learning
- continual-learning
configs:
- config_name: default
  data_files:
  - split: finetune
    path: data/finetune.parquet
  - split: stream
    path: data/stream.parquet
- config_name: full
  data_files:
  - split: train
    path: full/thd_temporal_full.parquet
---

# THD-Temporal: a 23-year Turkish news stream for online / continual text classification

**THD-Temporal** is a chronologically ordered stream of **{N_BENCH:,} Turkish news articles and columns published on
Milliyet.com between 1997 and 2019**, labelled with **10 news categories**. It is designed for studying
**concept drift, online (stream) learning and continual learning** of Turkish language models: the vocabulary and the
category distribution change strongly over the two decades (e.g. *Teknoloji-Bilim* is almost absent before 2008 and
reaches >30% of the articles in 2017–2018; the shares of *Ekonomi* and *Dünya* shrink while *Gündem* grows).

> **Credit.** THD-Temporal is a cleaned, re-documented derivative of the
> **Turkish News Corpus (Türkçe Haber Derlemi)** collected by **Ömer Gözüaçık** for the *Açık Kaynak Hackathon
> Programı 2020*: <https://github.com/ogozuacik/turkce-haber-derlemi> (MIT License, © 2020 Ömer Gözüaçık).
> All article selection and the original 12-class labelling are his work; please cite the original corpus
> (see *Citation*) whenever you use this dataset.

**Links:** code and benchmark: <{GITHUB_URL}> · Hugging Face: <https://huggingface.co/datasets/{HF_ID}> ·
Kaggle: <https://www.kaggle.com/datasets/{KAGGLE_ID}>

## What THD-Temporal adds to the original corpus

| Step | Detail |
|---|---|
| Article selection & labels | Taken unchanged from the original `filtrelenmis_temizlenmis_derlem.csv.gz` (81,407 articles, 12 categories). |
| Original text restored | The filtered file is lower-cased with numbers/punctuation removed; the original title, summary and body are restored from the raw scrape (`milliyet_derlem.csv.gz`) by joining on `link` (100% match). |
| Dates repaired | In the filtered file, day and month are swapped for every date with day ≤ 12 ({N_DATE_FIX:,} articles); dates are re-parsed from the raw `dd.mm.yyyy` strings. |
| 10-class label | `Ege` (regional edition), `Pazar` and `Cumartesi` (weekend supplements) are not topical and are merged into **`Diğer`**; the original 12-class `category` and the Milliyet section `category_original` are kept. |
| Quality flags | Excluded from the benchmark but kept, flagged, in the `full` config: 19 invalid dates (1899); {N_2020:,} articles whose only date is the June/July 2020 crawl date (93% sports, mostly from the Skorer site; article ids indicate publication in roughly 2009–2019, exact dates unknown; `exclusion = out_of_range_year`, their `date` is the crawl date); {N_DUP:,} exact duplicate articles. |
| Benchmark split | Chronological: **`finetune`** = 1997–2007 ({N_FT:,}), **`stream`** = 2008–2019 ({N_ST:,}), sorted by date. |

## Label distribution

{LABEL_TABLE}

Yearly volume: min {Y_MIN:,} ({Y_MIN_YEAR}), max {Y_MAX:,} ({Y_MAX_YEAR}), mean {Y_MEAN:,.0f} articles. About {COL_SHARE:.0%}
of the items are opinion columns (`is_column`).

## Columns

| column | description |
|---|---|
| `id` | stable identifier (`thd-<row of the original filtered file>`) |
| `date`, `year` | publication date (YYYY-MM-DD, repaired) and year |
| `title`, `summary`, `news` | original headline, summary (repeats the title for 17.5% of the items: 30% of `finetune`, 0.2% of `stream`) and full text |
| `label` | 10-class target: Diğer, Dünya, Ekonomi, Gündem, Kültür-Sanat, Magazin, Siyaset, Spor, Teknoloji-Bilim, Yaşam |
| `category` | original 12-class label of the source corpus |
| `category_original` | Milliyet section / sub-category ({N_CATORIG_BENCH} values in the benchmark, {N_CATORIG_FULL} in the `full` config) |
| `link`, `url` | relative Milliyet path and full URL |
| `is_column`, `author` | opinion column flag and columnist slug |
| `date_repaired` | the date in the source file had day/month swapped |
| `split` | `finetune` (1997–2007) or `stream` (2008–2019) |

The `full` config additionally has `in_benchmark` and `exclusion` (`invalid_date`, `out_of_range_year`, `duplicate`).

## Usage

```python
from datasets import load_dataset

# splits: finetune (1997-2007) and stream (2008-2019), both in chronological order
ds = load_dataset("{HF_ID}")
stream = ds["stream"].to_pandas()
```

```python
import kagglehub, pandas as pd
path = kagglehub.dataset_download("{KAGGLE_ID}")
df = pd.read_parquet(f"{path}/thd_temporal.parquet")
```

Model input used in our experiments: `title + ". " + summary`; the summary is skipped when it repeats the title, and the
`". "` becomes a single space when the title already ends in `. ! ? … : ; " '`.

## Benchmark: on-the-fly adaptation (OFA) of Turkish LLMs

The reference implementation and results (six Turkish encoders × {Static, OFA with AdamW, OFA with Adaptimizer},
pretrained and fine-tuned settings, prequential evaluation) are in <{GITHUB_URL}>. The protocol follows
LLM-OFA (Ghahramanian et al., CIKM 2025), which introduced the English counterpart 1M-News.

## Limitations

* Single source (Milliyet.com); labels are newspaper sections, so some classes (Gündem, Diğer) are broad.
* Strong class imbalance and abrupt label-distribution changes are intentional properties of the stream, but should be
  kept in mind when comparing accuracy and macro-F1.
* Article texts belong to their publisher; as in the original corpus they are shared for research purposes.

## Citation

Please cite the original corpus and, if you use the benchmark protocol, LLM-OFA:

```bibtex
@misc{gozuacik2020turkcehaberderlemi,
  author       = {G{\"o}z{\"u}a{\c{c}}{\i}k, {\"O}mer},
  title        = {T{\"u}rk{\c{c}}e Haber Derlemi ve Haberlerin Kategorik S{\i}n{\i}fland{\i}r{\i}lmas{\i}},
  year         = {2020},
  howpublished = {\url{https://github.com/ogozuacik/turkce-haber-derlemi}}
}
@inproceedings{ghahramanian2025ofa,
  author    = {Pouya Ghahramanian and Sepehr Bakhshi and Fazli Can},
  title     = {LLM-OFA: On-the-Fly Adaptation of Large Language Models to Address Temporal Drift Across Two Decades of News},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  year      = {2025},
  doi       = {10.1145/3746252.3760846}
}
```

## Acknowledgements

Prepared within TÜBİTAK 1001 project no. **125E060**. We thank Ömer Gözüaçık for collecting and openly sharing the
original corpus.

## Türkçe özet

THD-Temporal, Ömer Gözüaçık tarafından derlenip açık kaynak olarak paylaşılan **Türkçe Haber Derlemi**'nden
(Milliyet.com, 1997–2019) türetilmiş, zaman sırasına dizilmiş {N_BENCH:,} haberlik bir akıştır. Orijinal derlemdeki
haber seçimi ve etiketler korunmuş; haberlerin özgün (büyük-küçük harf ve noktalama içeren) metinleri geri
kazandırılmış, gün/ay karışıklığı içeren tarihler düzeltilmiş, bölgesel/hafta sonu ekleri (Ege, Pazar, Cumartesi)
"Diğer" sınıfında birleştirilerek 10 sınıflı bir etiket oluşturulmuş ve 1997–2007 (ince-ayar) / 2008–2019 (akış)
şeklinde zamansal bir bölme tanımlanmıştır. Veri kümesi, Türkçe dil modellerinin kavram kayması altında çevrimiçi
(akış) ve sürekli öğrenme başarımlarının incelenmesi için hazırlanmıştır.
