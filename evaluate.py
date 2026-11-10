
import pandas as pd
import json
import re
import string
import time
import os
import mlflow
import transformers
import torch

from collections import Counter
from transformers import pipeline


# --------------------------------------------------
# 1. CONNECT TO MLFLOW
# --------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("QA_Champion_Challenger")


# --------------------------------------------------
# 2. LOAD EVALUATION DATASET
# --------------------------------------------------

df = pd.read_csv("data/eval_inputs.csv")

os.makedirs("outputs", exist_ok=True)

print("Dataset loaded successfully!")
print("Total examples:", len(df))


# --------------------------------------------------
# 3. TEXT NORMALIZATION
# --------------------------------------------------

def normalize_text(text):

    text = text.lower()

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

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

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
# 6. EVALUATE AND TRACK ONE MODEL
# --------------------------------------------------

def evaluate_model(model_id, model_name, role, run_name):

    print("\nLoading model:", model_name)

    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):

        # Load pretrained QA model
        qa = pipeline(
            "question-answering",
            model=model_id,
            device=-1
        )

        predictions = []

        total_time = 0

        # Evaluate all examples
        for _, row in df.iterrows():

            question = row["question"]
            context = row["context"]
            reference = row["reference_answer"]

            start = time.perf_counter()

            result = qa(
                question=question,
                context=context
            )

            end = time.perf_counter()

            inference_time = end - start

            total_time += inference_time

            prediction = result["answer"]

            em = exact_match(prediction, reference)

            f1 = calculate_f1(prediction, reference)

            predictions.append({

                "id": row["id"],
                "question": question,
                "reference_answer": reference,
                "predicted_answer": prediction,
                "confidence": result["score"],
                "exact_match": em,
                "f1_score": f1,
                "inference_time_seconds": inference_time

            })

        # Convert predictions into DataFrame
        results_df = pd.DataFrame(predictions)

        # Calculate evaluation metrics
        em_score = results_df["exact_match"].mean() * 100

        f1_score = results_df["f1_score"].mean() * 100

        avg_latency = (total_time / len(df)) * 1000

        # Save prediction file
        output_file = f"outputs/predictions_{role}.csv"

        results_df.to_csv(output_file, index=False)

        # ------------------------------------------
        # LOG PARAMETERS
        # ------------------------------------------

        mlflow.log_params({

            "model_name": model_name,
            "model_id": model_id,
            "task": "question-answering",
            "role": role,
            "device": "CPU",
            "evaluation_samples": len(df),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__

        })

        # ------------------------------------------
        # LOG TAGS
        # ------------------------------------------

        mlflow.set_tags({

            "project": "QA_Champion_Challenger",
            "model_family": model_name,
            "dataset": "eval_inputs.csv"

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
            "data/eval_inputs.csv",
            artifact_path="dataset"
        )

        # Display results
        print("\nResults:", model_name)

        print("Exact Match:", round(em_score, 2), "%")

        print("F1 Score:", round(f1_score, 2), "%")

        print("Average Latency:", round(avg_latency, 2), "ms")

        print("Total Inference Time:", round(total_time, 2), "seconds")

        print("MLflow run:", run_name)

        return {

            "model": model_name,
            "model_id": model_id,
            "exact_match": round(em_score, 2),
            "f1_score": round(f1_score, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "total_inference_s": round(total_time, 4)

        }


# --------------------------------------------------
# 7. EVALUATE MINILM
# --------------------------------------------------

model_a = evaluate_model(

    "deepset/minilm-uncased-squad2",
    "MiniLM",
    "model_A",
    "A_qa_minilm_actual"

)


# --------------------------------------------------
# 8. EVALUATE ROBERTA
# --------------------------------------------------

model_b = evaluate_model(

    "deepset/roberta-base-squad2",
    "RoBERTa",
    "model_B",
    "B_qa_roberta_actual"

)


# --------------------------------------------------
# 9. SAVE METRICS SUMMARY
# --------------------------------------------------

summary = {

    "evaluation_samples": len(df),

    "model_A": model_a,

    "model_B": model_b

}

with open("outputs/metrics_summary.json", "w") as file:

    json.dump(summary, file, indent=4)


# --------------------------------------------------
# 10. FINAL COMPARISON
# --------------------------------------------------

print("\n===================================")
print("FINAL MODEL COMPARISON")
print("===================================")

print(pd.DataFrame([model_a, model_b]).to_string(index=False))

print("\nEvaluation completed successfully!")

print("Predictions, metrics and MLflow runs saved!")
