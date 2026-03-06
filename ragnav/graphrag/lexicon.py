from __future__ import annotations

# Small built-in lexicons for "real paper" entity extraction.
# These are intentionally incomplete but cover common benchmark papers.

METRICS = {
    "accuracy",
    "f1",
    "bleu",
    "rouge",
    "mrr",
    "ndcg",
    "anls",
    "exact match",
    "em",
    "perplexity",
}

TASKS = {
    "question answering",
    "reading comprehension",
    "classification",
    "retrieval",
    "summarization",
    "natural language inference",
    "named entity recognition",
    "machine translation",
    "language modeling",
}

DATASETS = {
    # NLP
    "squad",
    "glue",
    "mnli",
    "multinli",
    "mrpc",
    "sst-2",
    "sst2",
    "qqp",
    "qnli",
    "rte",
    "cola",
    "wmt14",
    "cnn/dailymail",
    "c4",
    # CV
    "imagenet",
    "cifar-10",
    "cifar10",
    "cifar-100",
    "coco",
}

MODELS = {
    "bert",
    "roberta",
    "t5",
    "gpt-2",
    "gpt2",
    "gpt-3",
    "llama",
    "resnet",
    "vit",
    "clip",
}

METHOD_CUES = {
    "baseline",
    "ablation",
    "pretraining",
    "fine-tuning",
    "finetuning",
    "distillation",
}


def normalize_key(s: str) -> str:
    return (s or "").strip().lower()


def contains_any(text: str, phrases: set[str]) -> bool:
    low = (text or "").lower()
    return any(p in low for p in phrases)

