# LLM-OFA-TR: On-the-Fly Adaptation of Turkish Language Models on a 23-Year News Stream

This repository is the Turkish counterpart of [LLM-OFA](https://github.com/PouyaGhahramanian/LLM-OFA)
(CIKM 2025). It provides

* **THD-Temporal**, a cleaned, chronologically ordered stream of **78,171 Turkish news articles (Milliyet.com,
  1997–2019, 10 categories)** built from the *Türkçe Haber Derlemi* of **Ömer Gözüaçık**, and
* a reference implementation of **online (stream) text classification with Turkish transformer encoders**: the model
  predicts each incoming batch, receives the true labels, and is updated on the fly (OFA), optionally with
  **Adaptimizer**, our fast/slow-weight optimizer for temporal drift.

It is meant as a starting point for anyone working on **Turkish online / continual text classification under concept
drift**.

## Dataset: THD-Temporal

> **Credit.** THD-Temporal is derived from the **Turkish News Corpus (Türkçe Haber Derlemi)** collected and openly
> shared by **Ömer Gözüaçık** (Açık Kaynak Hackathon Programı 2020):
> <https://github.com/ogozuacik/turkce-haber-derlemi>, MIT License, © 2020 Ömer Gözüaçık.
> The article selection and the original labels are his work; please cite the original corpus when using the data.

`scripts/prepare_thd_temporal.py` rebuilds the dataset from the two original files (downloaded automatically):

| Step | Detail |
|---|---|
| Article selection, labels | unchanged from `filtrelenmis_temizlenmis_derlem.csv.gz` (81,407 articles, 12 categories) |
| Original text | title / summary / body restored from the raw scrape `milliyet_derlem.csv.gz` (joined on `link`, 100% match); the filtered file is lower-cased without digits and punctuation |
| Dates | day/month swap in the filtered file repaired for 29,427 articles (dates re-parsed from raw `dd.mm.yyyy`) |
| Labels | 10 classes: `Ege` (regional edition), `Pazar`, `Cumartesi` (weekend supplements) merged into `Diğer` |
| Exclusions (flagged) | 19 invalid dates (1899), 2,247 articles of the partial year 2020, 970 exact duplicates |
| Split | `finetune` 1997–2007 (45,384) · `stream` 2008–2019 (32,787) |

| label | finetune (1997–2007) | stream (2008–2019) |
|---|---:|---:|
| Gündem | 9,502 | 11,515 |
| Siyaset | 9,325 | 5,193 |
| Ekonomi | 8,701 | 2,456 |
| Diğer | 6,010 | 2,601 |
| Spor | 6,335 | 2,213 |
| Dünya | 2,971 | 1,814 |
| Magazin | 1,001 | 2,779 |
| Teknoloji-Bilim | 54 | 3,042 |
| Yaşam | 770 | 832 |
| Kültür-Sanat | 715 | 342 |

The stream drifts strongly: `Teknoloji-Bilim` is practically absent before 2008 and makes up 33–42% of all articles
in 2017–2018, the shares of `Ekonomi`, `Spor` and `Dünya` shrink, `Gündem` grows (`figures/label_drift.png`), and the
vocabulary of each class changes over time (`figures/wordclouds_ekonomi.png`).

**Load it** in one line:

```python
from datasets import load_dataset
ds = load_dataset("p0uy4/thd-temporal")   # splits: finetune (1997-2007), stream (2008-2019)
```

```python
import kagglehub, pandas as pd
df = pd.read_parquet(kagglehub.dataset_download("pooyagh/thd-temporal") + "/thd_temporal.parquet")
```

or rebuild it from the original corpus (downloaded automatically):

```bash
python scripts/prepare_thd_temporal.py --orig-dir data/original --out-dir data
```

`load_dataset("p0uy4/thd-temporal", "full")` returns all 81,407 source articles with the exclusion flags.

## Method and protocol

Same protocol as LLM-OFA (CIKM 2025):

* **Settings.** *pretrained*: the Hugging Face checkpoint, classification head calibrated with one pass over a random
  1% of the stream (removed from the stream afterwards); *fine-tuned*: trained on 1997–2007 (90/10 split, 3 epochs,
  lr 5e-5, best epoch by validation loss).
* **Configurations.** *Static* (no updates) · *OFA* (test-then-train, AdamW, lr 2e-5) · *OFA-Adapt* (test-then-train
  with Adaptimizer: lr 2e-5, squared-gradient window w = 10, slow-weight EMA rate ρ = 0.1; predictions with the slow
  weights; drift score = cosine distance between fast and slow weights; slow weights re-synchronised when the drift
  score exceeds twice its running mean).
* **Input.** `title + ". " + summary`, max 160 sub-word tokens.
* **Metrics.** accuracy, macro-F1 over the whole stream; prequential accuracy over consecutive windows of 200 items
  (and the worst window).

| family | Turkish checkpoint | batch size |
|---|---|---:|
| BERT | [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased) (BERTurk) | 32 |
| DistilBERT | [`dbmdz/distilbert-base-turkish-cased`](https://huggingface.co/dbmdz/distilbert-base-turkish-cased) | 64 |
| ELECTRA | [`dbmdz/electra-base-turkish-cased-discriminator`](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator) | 16 |
| ALBERT | [`loodos/albert-base-turkish-uncased`](https://huggingface.co/loodos/albert-base-turkish-uncased) | 16 |
| RoBERTa | [`TURKCELL/roberta-base-turkish-uncased`](https://huggingface.co/TURKCELL/roberta-base-turkish-uncased) | 32 |
| DeBERTa | [`microsoft/mdeberta-v3-base`](https://huggingface.co/microsoft/mdeberta-v3-base) (multilingual; no Turkish DeBERTa exists) | 8 |

Uncased checkpoints receive Turkish-aware lower-casing (`I → ı`, `İ → i`); their tokenizers' built-in lower-casing
turns `İ` into an unknown token.

## Results

<!-- RESULTS -->

## Reproduce

```bash
pip install -r requirements.txt          # torch >= 2.6 (needed to load the .bin checkpoints safely)
python scripts/prepare_thd_temporal.py --orig-dir data/original --out-dir data
python scripts/run_all.py --gpus 0 1      # 6 fine-tuning jobs + 36 stream runs, resumable
python scripts/aggregate.py               # tables -> results/
python scripts/plot_stream.py             # figures/stream_accuracy_drift.png
python scripts/plot_dataset.py            # figures/label_drift.png, figures/wordclouds_*.png
```

Single runs:

```bash
python scripts/finetune.py --model bert
python scripts/stream.py --model bert --setting finetuned --config ofa-adapt
python scripts/stream.py --model bert --setting pretrained --config static
```

Each run writes `results/stream/<setting>_<model>_<config>/` with `metrics.json`, `predictions.csv.gz`
(per-item predictions, usable for significance tests), `drift.csv.gz` (OFA-Adapt), `window_accuracy.txt` and a
classification report.

## Repository layout

```
src/ofa_tr/      data loading, Turkish model registry, Adaptimizer, plotting style
scripts/         dataset build, fine-tuning, stream evaluation, job runner, tables, figures, hub export
results/         per-run metrics and predictions, aggregated tables
figures/         figures used in the paper/report
hub/             dataset card used for the Hugging Face / Kaggle releases
```

## Availability

* **Dataset (Hugging Face):** <https://huggingface.co/datasets/p0uy4/thd-temporal>
* **Dataset (Kaggle):** <https://www.kaggle.com/datasets/pooyagh/thd-temporal>
* **Code, predictions and result tables:** this repository
* **Original corpus:** <https://github.com/ogozuacik/turkce-haber-derlemi> (Ömer Gözüaçık, MIT)

## Citation

```bibtex
@inproceedings{ghahramanian2025ofa,
  author    = {Pouya Ghahramanian and Sepehr Bakhshi and Fazli Can},
  title     = {LLM-OFA: On-the-Fly Adaptation of Large Language Models to Address Temporal Drift Across Two Decades of News},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  year      = {2025},
  doi       = {10.1145/3746252.3760846}
}
@misc{gozuacik2020turkcehaberderlemi,
  author       = {G{\"o}z{\"u}a{\c{c}}{\i}k, {\"O}mer},
  title        = {T{\"u}rk{\c{c}}e Haber Derlemi ve Haberlerin Kategorik S{\i}n{\i}fland{\i}r{\i}lmas{\i}},
  year         = {2020},
  howpublished = {\url{https://github.com/ogozuacik/turkce-haber-derlemi}}
}
```

## Acknowledgements

This work is supported by TÜBİTAK (The Scientific and Technological Research Council of Türkiye) under grant
no. 125E060. We thank Ömer Gözüaçık for collecting and openly sharing the Turkish News Corpus.

## License

Code: MIT (see `LICENSE`). Data: derived from the MIT-licensed Turkish News Corpus (see `data/LICENSE-THD.txt`);
article texts were originally published on Milliyet.com and are redistributed, as in the original corpus, for research.

## Türkçe özet

Bu depo, CIKM 2025'te yayımlanan LLM-OFA çalışmamızın Türkçe karşılığıdır. Ömer Gözüaçık'ın derleyip açık kaynak
olarak paylaştığı Türkçe Haber Derlemi'nden (Milliyet.com, 1997–2019) türettiğimiz **THD-Temporal** veri kümesini ve
Türkçe dil modellerinin (BERTurk, DistilBERTurk, ELECTRA, ALBERT, RoBERTa, mDeBERTa) zaman içinde değişen haber
akışında çevrimiçi olarak (on-the-fly) uyarlanmasına yönelik kodları içermektedir. Akıştaki her haber grubu önce
sınıflandırılmakta, ardından gerçek etiketleriyle model güncellenmektedir (OFA); Adaptimizer ise hızlı/yavaş ağırlık
mekanizmasıyla kayma anlarında hızlı ve kararlı uyum sağlamaktadır. TÜBİTAK 125E060 numaralı proje kapsamında
geliştirilmiştir.
