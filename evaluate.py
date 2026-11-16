
import os
import re
import json
import time
import string
import random
import platform
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import transformers
import torch

from transformers import pipeline


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "QA_Champion_Challenger"

STUDENT_USN = "1MS25CI073"
SEED = 42

DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = "cuda:0" if DEVICE == 0 else "cpu"

DATA_PATH = "data/eval_inputs.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("MLflow connected successfully!")
print("Device:", DEVICE_NAME)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

# keep_default_na=False preserves intentionally empty answers.
df = pd.read_csv(DATA_PATH, keep_default_na=False)

required_columns = [
    "id",
    "context",
    "question",
    "reference_answer"
]

if not all(col in df.columns for col in required_columns):
    raise ValueError("Dataset is missing required columns.")

# ID, context and question must not be empty.
for col in ["id", "context", "question"]:
    if df[col].astype(str).str.strip().eq("").any():
        raise ValueError(f"Dataset contains empty values in {col}.")

# Empty reference answers are allowed.
df["reference_answer"] = (
    df["reference_answer"].fillna("").astype(str)
)

if df["id"].duplicated().any():
    raise ValueError("Dataset contains duplicate IDs.")

print("\nDataset loaded successfully!")
print("Total examples:", len(df))
print(
    "Unanswerable examples:",
    (df["reference_answer"].str.strip() == "").sum()
)


# --------------------------------------------------
# 3. TEXT NORMALIZATION
# --------------------------------------------------

def normalize_text(text):

    text = str(text).lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    text = re.sub(r"\b(a|an|the)\b", " ", text)

    text = " ".join(text.split())

    return text


# --------------------------------------------------
# 4. EXACT MATCH
# --------------------------------------------------

def exact_match(prediction, reference):

    pred = normalize_text(prediction)
    ref = normalize_text(reference)

    return int(pred == ref)


# --------------------------------------------------
# 5. F1 SCORE
# --------------------------------------------------

def calculate_f1(prediction, reference):

    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    # Both prediction and reference are empty.
    if not pred_tokens and not ref_tokens:
        return 1.0

    # Only one is empty.
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)

    common_count = sum(common.values())

    if common_count == 0:
        return 0.0

    precision = common_count / len(pred_tokens)
    recall = common_count / len(ref_tokens)

    return (
        2 * precision * recall
    ) / (precision + recall)


# --------------------------------------------------
# 6. EVALUATE ONE MODEL
# --------------------------------------------------

def evaluate_model(model_id, model_name, role, run_name):

    print("\nLoading model:", model_name)

    with mlflow.start_run(run_name=run_name) as run:

        run_id = run.info.run_id

        qa = pipeline(
            "question-answering",
            model=model_id,
            device=DEVICE
        )

        predictions = []
        total_time = 0

        for _, row in df.iterrows():

            # Synchronize GPU for accurate timing.
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            result = qa(
                question=row["question"],
                context=row["context"],
                handle_impossible_answer=True
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            inference_time = time.perf_counter() - start

            total_time += inference_time

            prediction = result["answer"]
            reference = row["reference_answer"]

            em = exact_match(prediction, reference)

            f1 = calculate_f1(prediction, reference)

            predictions.append({

                "id": row["id"],
                "question": row["question"],
                "context": row["context"],
                "reference_answer": reference,
                "predicted_answer": prediction,
                "confidence": float(result["score"]),
                "exact_match": em,
                "f1_score": f1,
                "inference_time_seconds": inference_time,
                "reference_unanswerable": reference.strip() == "",
                "prediction_unanswerable": prediction.strip() == ""

            })

        results_df = pd.DataFrame(predictions)

        em_score = results_df["exact_match"].mean() * 100
        f1_score = results_df["f1_score"].mean() * 100

        avg_latency = (
            total_time / len(df)
        ) * 1000

        output_file = (
            f"{OUTPUT_DIR}/predictions_{role}.csv"
        )

        results_df.to_csv(output_file, index=False)

        # ------------------------------------------
        # LOG PARAMETERS
        # ------------------------------------------

        mlflow.log_params({

            "model_name": model_name,
            "hf_model_id": model_id,
            "task": "question-answering",
            "role": role,
            "eval_samples": len(df),
            "seed": SEED,
            "device": DEVICE_NAME,
            "batch_size": 1,
            "handle_impossible_answer": True

        })

        # ------------------------------------------
        # LOG TAGS
        # ------------------------------------------

        library_versions = {
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "mlflow": mlflow.__version__,
            "pandas": pd.__version__
        }

        mlflow.set_tags({

            "student_usn": STUDENT_USN,
            "dataset_name": "eval_inputs.csv",
            "library_versions": json.dumps(library_versions),
            "model_name": model_name,
            "project": EXPERIMENT_NAME

        })

        # ------------------------------------------
        # LOG METRICS
        # ------------------------------------------

        mlflow.log_metrics({

            "exact_match_percent": float(em_score),
            "f1_score_percent": float(f1_score),
            "average_latency_ms": float(avg_latency),
            "total_inference_seconds": float(total_time)

        })

        # ------------------------------------------
        # LOG ARTIFACTS
        # ------------------------------------------

        mlflow.log_artifact(
            output_file,
            artifact_path="predictions"
        )

        mlflow.log_artifact(
            DATA_PATH,
            artifact_path="dataset"
        )

        print("\nResults:", model_name)
        print("Exact Match:", round(em_score, 2), "%")
        print("F1 Score:", round(f1_score, 2), "%")
        print("Average Latency:", round(avg_latency, 2), "ms")
        print("Total Inference Time:", round(total_time, 4), "seconds")
        print("MLflow Run ID:", run_id)

        # Release model resources.
        del qa

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {

            "model": model_name,
            "model_id": model_id,
            "exact_match": round(em_score, 2),
            "f1_score": round(f1_score, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "total_inference_s": round(total_time, 4),
            "run_id": run_id

        }, results_df


# --------------------------------------------------
# 7. EVALUATE MINILM
# --------------------------------------------------

model_a, predictions_a = evaluate_model(

    "deepset/minilm-uncased-squad2",
    "MiniLM",
    "champion_candidate",
    "A_qa_minilm"

)


# --------------------------------------------------
# 8. EVALUATE ROBERTA
# --------------------------------------------------

model_b, predictions_b = evaluate_model(

    "deepset/roberta-base-squad2",
    "RoBERTa",
    "challenger_candidate",
    "B_qa_roberta"

)


# --------------------------------------------------
# 9. SAVE METRICS SUMMARY
# --------------------------------------------------

summary = {

    "evaluation_samples": len(df),
    "student_usn": STUDENT_USN,
    "seed": SEED,
    "device": DEVICE_NAME,
    "model_A": model_a,
    "model_B": model_b

}

summary_path = f"{OUTPUT_DIR}/metrics_summary.json"

with open(summary_path, "w") as file:

    json.dump(summary, file, indent=4)


# --------------------------------------------------
# 10. CREATE EVALUATION PLOTS
# --------------------------------------------------

models = ["MiniLM", "RoBERTa"]

# Plot 1: EM and F1 comparison

em_values = [
    model_a["exact_match"],
    model_b["exact_match"]
]

f1_values = [
    model_a["f1_score"],
    model_b["f1_score"]
]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    em_values,
    width,
    label="Exact Match"
)

plt.bar(
    x + width / 2,
    f1_values,
    width,
    label="F1 Score"
)

plt.xticks(x, models)
plt.ylabel("Score (%)")
plt.title("Question Answering Accuracy Comparison")
plt.ylim(0, 110)
plt.legend()
plt.tight_layout()

accuracy_plot = f"{OUTPUT_DIR}/accuracy_comparison.png"

plt.savefig(accuracy_plot)
plt.close()


# Plot 2: Average inference latency

latency_values = [
    model_a["avg_latency_ms"],
    model_b["avg_latency_ms"]
]

plt.figure(figsize=(8, 5))

plt.bar(models, latency_values)

plt.ylabel("Average Latency (ms)")
plt.title("Model Inference Latency Comparison")
plt.tight_layout()

latency_plot = f"{OUTPUT_DIR}/latency_comparison.png"

plt.savefig(latency_plot)
plt.close()


# Plot 3: Confidence score distribution

plt.figure(figsize=(8, 5))

plt.hist(
    predictions_a["confidence"],
    bins=8,
    alpha=0.6,
    label="MiniLM"
)

plt.hist(
    predictions_b["confidence"],
    bins=8,
    alpha=0.6,
    label="RoBERTa"
)

plt.xlabel("Confidence Score")
plt.ylabel("Number of Predictions")
plt.title("Question Answering Confidence Distribution")
plt.legend()
plt.tight_layout()

confidence_plot = f"{OUTPUT_DIR}/confidence_distribution.png"

plt.savefig(confidence_plot)
plt.close()


# --------------------------------------------------
# 11. LOG SUMMARY ARTIFACTS TO BOTH MODEL RUNS
# --------------------------------------------------

for run_id in [model_a["run_id"], model_b["run_id"]]:

    with mlflow.start_run(run_id=run_id):

        mlflow.log_artifact(
            summary_path,
            artifact_path="summary"
        )

        mlflow.log_artifact(
            accuracy_plot,
            artifact_path="plots"
        )

        mlflow.log_artifact(
            latency_plot,
            artifact_path="plots"
        )

        mlflow.log_artifact(
            confidence_plot,
            artifact_path="plots"
        )


# --------------------------------------------------
# 12. FINAL COMPARISON
# --------------------------------------------------

print("\n===================================")
print("FINAL MODEL COMPARISON")
print("===================================")

comparison_df = pd.DataFrame([
    model_a,
    model_b
])

print(comparison_df.to_string(index=False))

print("\nEvaluation completed successfully!")
print("Metrics summary saved.")
print("Three evaluation plots saved.")
print("MLflow artifacts updated.")
print("Evaluation device:", DEVICE_NAME)
