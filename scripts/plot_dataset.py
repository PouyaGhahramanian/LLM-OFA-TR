#!/usr/bin/env python
"""Dataset figures for THD-Temporal.

  figures/label_drift.png  yearly share of each class 1997-2019 (small multiples; natural label drift)
  figures/wordclouds_<label>.png  TF-IDF-weighted word clouds of one class in four periods
  results/label_share_by_year.csv  table view of the label-drift figure
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ofa_tr.data import load_thd, turkish_lower  # noqa: E402
from ofa_tr.plotting import ACCENT, BLUE_RAMP, INK, SURFACE, apply_style, plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PERIODS = [(1997, 2002), (2003, 2007), (2008, 2013), (2014, 2019)]
# compact Turkish stop-word list (function words, auxiliaries, frequent news fillers)
STOP = set("""acaba ama ancak artık aslında az bana bazı belki ben beni benim beri bile bir birçok biri birkaç birşey
biz bize bizi bizim bu buna bunda bundan bunlar bunları bunların bunu bunun burada böyle böylece çok çünkü da daha
dahi de defa değil diğer diye dolayı en fakat gibi göre hala hangi hatta hem hep hepsi her herhangi herkes hiç hiçbir
için ile ilgili ise işte itibaren kadar karşın kendi kendine kez ki kim kime kimi kimse mi mı mu mü na nasıl ne neden
nerde nerede nereye niçin niye o olan olarak oldu olduğu olduğunu olmak olması olmayan olmaz olsa olsun olup olur olursa
oluyor on ona ondan onlar onları onların onu onun öyle önce sadece sanki şey şimdi şu şuna şunda şundan şunu tarafından
tüm üzere var vardı ve veya ya yani yapılan yaptı yeni yine yoksa zaten ayrıca bunlardan sonra son ilk büyük iki üç
milyon milyar yıl yılı yılında yüzde lira dedi söyledi etti eden etmek edildi oldu olacak değildir gün bugün dün yarın
milliyet gazetesi gazete com www olayların içinden haber haberi dakika açıkladı geldi gelen gelecek bin""".split())


def label_drift(df):
    share = pd.crosstab(df["year"], df["label"], normalize="index") * 100
    order = df["label"].value_counts().index.tolist()
    share = share[order]
    share.to_csv(REPO / "results" / "label_share_by_year.csv", float_format="%.3f")
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.3), sharex=True)
    for ax, lab in zip(axes.flat, order):
        y = share[lab]
        ax.plot(share.index, y, color=ACCENT)
        ax.fill_between(share.index, y, 0, color=ACCENT, alpha=0.10, lw=0)
        ax.axvline(2007.5, color=INK["muted"], lw=0.7)
        ax.set_title(lab, loc="left", fontsize=8.5)
        ax.set_ylim(0, max(5, float(np.ceil(y.max() / 5) * 5)))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
        ax.set_xticks([1997, 2007, 2019])
        ax.tick_params(labelsize=7)
    # the vertical hairline marks the start of the stream (2008); explained in the caption
    fig.supylabel("Yıl içindeki pay (%)", fontsize=8, color=INK["secondary"])
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "label_drift.png", dpi=300)
    fig.savefig(REPO / "figures" / "label_drift.pdf")


def clean_for_terms(text):
    """Lower-case (Turkish rules) and drop suffixes after apostrophes: "Türkiye'nin" -> "türkiye"."""
    return re.sub(r"['’`´]\w*", " ", turkish_lower(text))


def wordclouds(df, label, top=60):
    """Class-based TF-IDF per period (each period = one document): w(t,p) = tf(t,p) * log(1 + A / f(t))."""
    from wordcloud import WordCloud
    sub = df[df["label"] == label]
    docs = [clean_for_terms(t) for t in sub["text"]]
    vec = CountVectorizer(token_pattern=r"(?u)\b[a-zçğıöşüâîû]{3,}\b", stop_words=sorted(STOP), min_df=3)
    x = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    masks = [((sub["year"] >= a) & (sub["year"] <= b)).to_numpy() for a, b in PERIODS]
    counts = np.vstack([np.asarray(x[m].sum(axis=0)).ravel() for m in masks])  # periods x terms
    tf = counts / counts.sum(axis=1, keepdims=True)
    idf = np.log(1 + counts.sum(axis=1).mean() / np.maximum(counts.sum(axis=0), 1))
    ctfidf = tf * idf
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    ramp = BLUE_RAMP

    def color_func(word, font_size, position, orientation, random_state=None, **kw):
        return ramp[min(len(ramp) - 1, int(font_size / 70 * len(ramp)))]

    rows = []
    for k, (ax, (a, b)) in enumerate(zip(axes, PERIODS)):
        mask = masks[k]
        w = ctfidf[k]
        idx = np.argsort(-w)[:top]
        freqs = {vocab[i]: float(w[i]) for i in idx if w[i] > 0}
        rows += [{"period": f"{a}-{b}", "term": t, "tfidf_sum": v} for t, v in freqs.items()]
        wc = WordCloud(width=600, height=420, background_color=SURFACE, color_func=color_func,
                       prefer_horizontal=1.0, random_state=0, margin=4, max_font_size=70).generate_from_frequencies(freqs)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_axis_off()
        ax.set_title(f"{a}-{b} (n={int(mask.sum())})", fontsize=8, color=INK["secondary"])
    fig.tight_layout()
    slug = re.sub(r"[^a-z]", "", turkish_lower(label).replace("ü", "u").replace("ş", "s").replace("ç", "c")
                  .replace("ğ", "g").replace("ı", "i").replace("ö", "o"))
    fig.savefig(REPO / "figures" / f"wordclouds_{slug}.png", dpi=300)
    pd.DataFrame(rows).to_csv(REPO / "results" / f"wordcloud_terms_{slug}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wordcloud-labels", nargs="+", default=["Ekonomi", "Gündem"])
    args = ap.parse_args()
    apply_style()
    (REPO / "figures").mkdir(exist_ok=True)
    df = load_thd()
    label_drift(df)
    for lab in args.wordcloud_labels:
        wordclouds(df, lab)
    print("saved figures")


if __name__ == "__main__":
    main()
