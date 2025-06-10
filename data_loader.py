from datasets import load_dataset
from transformers import LongformerTokenizerFast

tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")

def preprocess(example):
    # (Your preprocessing remains the same)
    inputs = tokenizer(
        example["article"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )
    targets = tokenizer(
        example["highlights"],
        truncation=True,
        padding="max_length",
        max_length=200,
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"],
    }

def load_data():
    # 1) Load only the first 10,000 examples
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:10000]")

    # 2) Split into 6k train, 2k validation, 2k test
    train_data = dataset.select(range(0, 6000)).map(
        preprocess, batched=True, remove_columns=["article", "highlights", "id"]
    )
    val_data = dataset.select(range(6000, 8000)).map(
        preprocess, batched=True, remove_columns=["article", "highlights", "id"]
    )

    """
    test_data = dataset.select(range(8000, 10000)).map(
        preprocess, batched=True, remove_columns=["article", "highlights", "id"]
    )
    """

    return train_data, val_data
