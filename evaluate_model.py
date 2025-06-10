from transformers import LongformerTokenizerFast, EncoderDecoderModel
from datasets import load_dataset
from rouge_score import rouge_scorer

# 1) Load the final model (can be used for inference)
model = EncoderDecoderModel.from_pretrained("./results/final_model")
tokenizer = LongformerTokenizerFast.from_pretrained("./results/final_model")

# 2) Load test set (CNN/DailyMail examples 70k–100k)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[70000:100000]")

# 3) Limit to first 100 examples (for speed—adjust as you wish)
test_dataset = dataset.select(range(100))

# 4) Initialize a ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

scores = {"rouge1": [], "rouge2": [], "rougeL": []}

# 5) Loop over test examples
for example in test_dataset:
    # Encode article
    input_ids = tokenizer.encode(
        example["article"],
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    # Generate a summary
    output_ids = model.generate(
        input_ids,
        max_length=200,
        no_repeat_ngram_size=3
    )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reference = example["highlights"]

    # Compute ROUGE scores
    rouge = scorer.score(reference, generated)
    scores["rouge1"].append(rouge["rouge1"].fmeasure)
    scores["rouge2"].append(rouge["rouge2"].fmeasure)
    scores["rougeL"].append(rouge["rougeL"].fmeasure)

# 6) Average each metric
avg_scores = {k: sum(v) / len(v) for k, v in scores.items()}

print("\n🧪 Evaluation Results (100 Samples):")
for metric, value in avg_scores.items():
    print(f"{metric}: {value:.4f}")
