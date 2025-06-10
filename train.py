from transformers import Seq2SeqTrainer
from model_setup import get_model
from data_loader import load_data
from config import training_args
import os

# 1) Load the “safe” encoder–decoder and tokenizer
model, tokenizer = get_model()

# 2) Load train/validation datasets (already tokenized)
train_data, val_data = load_data()

# 3) Create a Seq2SeqTrainer (it may pass num_items_in_batch,
#    but SafeEncoderDecoderModel will drop that kwarg)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# 4) Find last checkpoint (if any) under ./results/
last_checkpoint = None
if os.path.isdir("./results"):
    checkpoints = [
        os.path.join("./results", d)
        for d in os.listdir("./results")
        if d.startswith("checkpoint-")
    ]
    if checkpoints:
        last_checkpoint = sorted(
            checkpoints,
            key=lambda x: int(x.split("-")[-1])
        )[-1]

# 5) Train (or resume from last checkpoint)
trainer.train(resume_from_checkpoint=last_checkpoint)

# 6) Save final model and tokenizer
model.save_pretrained("./results/final_model")
tokenizer.save_pretrained("./results/final_model")
