
import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import torch

# ---------------- CONFIGURATION ----------------

TRACKING_URI = "http://127.0.0.1:5000"

REGISTERED_MODEL_NAME = "QA_Question_Answering"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@champion"

INPUT_FILE = "data/new_data.csv"
OUTPUT_FILE = "outputs/new_data_predictions.csv"

DEVICE = 0 if torch.cuda.is_available() else -1

# ---------------- LOAD CHAMPION DETAILS ----------------

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()

model_version = client.get_model_version_by_alias(
    REGISTERED_MODEL_NAME,
    "champion"
)

version_number = model_version.version
model_tags = model_version.tags

hf_model_id = model_tags.get(
    "hf_model_id", "Not available"
)

print("\n========== CHAMPION MODEL ==========")
print("Registered Model:", REGISTERED_MODEL_NAME)
print("Champion Version:", version_number)
print("Hugging Face Model:", hf_model_id)

# ---------------- LOAD NEW DATA ----------------

df = pd.read_csv(INPUT_FILE, keep_default_na=False)

required_columns = ["id", "context", "question"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

if df.empty:
    raise ValueError("new_data.csv contains no test examples.")

print("\nTotal new test examples:", len(df))

# ---------------- LOAD MODEL FROM MLFLOW ----------------

print("\nLoading Champion model from MLflow...")

model = mlflow.transformers.load_model(
    MODEL_URI,
    device=DEVICE
)

print("Model loaded successfully!")

# ---------------- MAKE PREDICTIONS ----------------

results = []

for _, row in df.iterrows():

    prediction = model(
        question=row["question"],
        context=row["context"]
    )

    answer = prediction.get("answer", "")
    confidence = prediction.get("score", 0.0)

    results.append({
        "id": row["id"],
        "question": row["question"],
        "predicted_answer": answer,
        "confidence": confidence,
        "model_version": version_number,
        "hf_model_id": hf_model_id
    })

    print("\n------------------------------")
    print("Test:", row["id"])
    print("Question:", row["question"])
    print("Predicted Answer:", answer)
    print("Confidence:", round(confidence, 3))

# ---------------- SAVE RESULTS ----------------

os.makedirs("outputs", exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n====================================")
print("PREDICTION COMPLETED SUCCESSFULLY!")
print("Champion Version:", version_number)
print("Total Predictions:", len(results_df))
print("Saved to:", OUTPUT_FILE)
print("====================================")
