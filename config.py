from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
    fp16=False,
    remove_unused_columns=False
)