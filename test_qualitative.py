
import os
import gc
import torch
import pandas as pd
from transformers import pipeline

# ---------------- CONFIGURATION ----------------

INPUT_FILE = "data/qualitative_candidates.csv"
OUTPUT_FILE = "outputs/qualitative_candidate_results.csv"

MODELS = {
    "RoBERTa": "deepset/roberta-base-squad2",
    "MiniLM": "deepset/minilm-uncased-squad2"
}

DEVICE = 0 if torch.cuda.is_available() else -1

# ---------------- LOAD DATA ----------------

df = pd.read_csv(INPUT_FILE, keep_default_na=False)

required = ["id", "context", "question", "reference_answer"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

if df.empty:
    raise ValueError("Candidate dataset is empty.")

print("Candidate examples:", len(df))
print("Device:", "GPU" if DEVICE == 0 else "CPU")

all_results = {}

# ---------------- EVALUATE MODELS ----------------

for model_name, model_id in MODELS.items():

    print(f"\nEvaluating {model_name}: {model_id}")

    qa = pipeline(
        "question-answering",
        model=model_id,
        device=DEVICE
    )

    predictions = []

    for _, row in df.iterrows():

        result = qa(
            question=row["question"],
            context=row["context"],
            handle_impossible_answer=True
        )

        predictions.append({
            "id": row["id"],
            "prediction": result["answer"],
            "confidence": result["score"]
        })

        print(
            f"ID {row['id']}: {result['answer']} "
            f"(confidence: {result['score']:.3f})"
        )

    all_results[model_name] = pd.DataFrame(predictions)

    del qa
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------- COMPARE PREDICTIONS ----------------

comparison = df.copy()

for model_name in MODELS:
    model_results = all_results[model_name]

    comparison = comparison.merge(
        model_results,
        on="id",
        how="left"
    )

    comparison.rename(
        columns={
            "prediction": f"{model_name.lower()}_prediction",
            "confidence": f"{model_name.lower()}_confidence"
        },
        inplace=True
    )

comparison["predictions_differ"] = (
    comparison["roberta_prediction"].str.strip().str.lower()
    != comparison["minilm_prediction"].str.strip().str.lower()
)

os.makedirs("outputs", exist_ok=True)

comparison.to_csv(OUTPUT_FILE, index=False)

# ---------------- DISPLAY DIFFERENCES ----------------

different = comparison[comparison["predictions_differ"]]

print("\n========== COMPARISON SUMMARY ==========")
print("Total examples:", len(comparison))
print("Different predictions:", len(different))

for _, row in different.iterrows():
    print("\nID:", row["id"])
    print("Question:", row["question"])
    print("Reference:", row["reference_answer"])
    print("RoBERTa:", row["roberta_prediction"])
    print("MiniLM:", row["minilm_prediction"])

print("\nResults saved to:", OUTPUT_FILE)
print("========================================")
