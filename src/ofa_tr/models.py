"""Turkish counterparts of the six encoder families used in LLM-OFA (CIKM 2025).

| key        | English model (LLM-OFA)          | Turkish model used here                           |
|------------|----------------------------------|---------------------------------------------------|
| bert       | bert-base-uncased                | dbmdz/bert-base-turkish-cased (BERTurk)           |
| distilbert | distilbert-base-uncased          | dbmdz/distilbert-base-turkish-cased (DistilBERTurk)|
| electra    | google/electra-base-discriminator| dbmdz/electra-base-turkish-cased-discriminator    |
| albert     | albert-base-v2                   | loodos/albert-base-turkish-uncased                |
| roberta    | roberta-base                     | TURKCELL/roberta-base-turkish-uncased             |
| deberta    | microsoft/deberta-base           | microsoft/mdeberta-v3-base (no Turkish DeBERTa)   |

Uncased checkpoints get Turkish-aware lower-casing (I -> ı, İ -> i): the ALBERT tokenizer's own lower-casing maps
"İ" to "i" + <unk> and "I" to "i", and the TURKCELL RoBERTa tokenizer (byte-level BPE) does no lower-casing at all
although its vocabulary was learned from lower-cased text.
Batch sizes follow LLM-OFA: 64 DistilBERT, 32 BERT/RoBERTa, 16 ALBERT/ELECTRA, 8 DeBERTa.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import ID2LABEL, LABEL2ID, turkish_lower

MODELS = {
    "bert": {"hf_id": "dbmdz/bert-base-turkish-cased", "batch_size": 32, "lowercase": False},
    "distilbert": {"hf_id": "dbmdz/distilbert-base-turkish-cased", "batch_size": 64, "lowercase": False},
    "electra": {"hf_id": "dbmdz/electra-base-turkish-cased-discriminator", "batch_size": 16, "lowercase": False},
    "albert": {"hf_id": "loodos/albert-base-turkish-uncased", "batch_size": 16, "lowercase": True},
    "roberta": {"hf_id": "TURKCELL/roberta-base-turkish-uncased", "batch_size": 32, "lowercase": True},
    "deberta": {"hf_id": "microsoft/mdeberta-v3-base", "batch_size": 8, "lowercase": False},
}


def prepare_texts(texts, model_key):
    return [turkish_lower(t) for t in texts] if MODELS[model_key]["lowercase"] else list(texts)


def load_model(model_key, checkpoint=None):
    """Load tokenizer + sequence-classification model (HF checkpoint or a local fine-tuned directory)."""
    source = checkpoint or MODELS[model_key]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(LABEL2ID), label2id=LABEL2ID, id2label=ID2LABEL)
    return tokenizer, model
