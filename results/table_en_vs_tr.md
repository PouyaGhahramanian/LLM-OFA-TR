| Ayar | Metrik | Dil | Static | OFA | OFA-Adapt | OFA − Static | OFA-Adapt − Static | OFA-Adapt − OFA |
|---|---|---|---|---|---|---|---|---|
| Önceden eğitilmiş | Doğruluk | İngilizce (1M-News) | 71.59 | 85.85 | 86.60 | +14.26 | +15.01 | +0.75 |
| Önceden eğitilmiş | Doğruluk | Türkçe (THD-Temporal) | 36.66 | 68.10 | 69.02 | +31.44 | +32.36 | +0.92 |
| Önceden eğitilmiş | Makro-F1 | İngilizce (1M-News) | 62.24 | 85.00 | 85.81 | +22.77 | +23.57 | +0.80 |
| Önceden eğitilmiş | Makro-F1 | Türkçe (THD-Temporal) | 7.20 | 54.92 | 56.01 | +47.72 | +48.81 | +1.08 |
| İnce-ayarlı | Doğruluk | İngilizce (1M-News) | 91.22 | 91.87 | 92.16 | +0.65 | +0.94 | +0.29 |
| İnce-ayarlı | Doğruluk | Türkçe (THD-Temporal) | 56.00 | 74.73 | 75.35 | +18.73 | +19.35 | +0.62 |
| İnce-ayarlı | Makro-F1 | İngilizce (1M-News) | 90.54 | 91.47 | 91.75 | +0.93 | +1.20 | +0.28 |
| İnce-ayarlı | Makro-F1 | Türkçe (THD-Temporal) | 45.92 | 66.87 | 67.46 | +20.95 | +21.54 | +0.59 |

*English values are copied from LLM-OFA (CIKM 2025) Table 2. In the released English runs the pretrained OFA/OFA-Adapt models started from an uncalibrated head on the full stream and the pretrained Static model was scored on the stream without its first 4,113 items; the Turkish runs follow the protocol as described in the paper (same calibrated start for all three configurations).*
