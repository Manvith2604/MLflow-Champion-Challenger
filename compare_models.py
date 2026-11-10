
import json
import mlflow

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("QA_Champion_Challenger")


# Load actual evaluation results
with open("outputs/metrics_summary.json", "r") as file:
    results = json.load(file)

model_a = results["model_A"]
model_b = results["model_B"]


# Define comparison rule
def select_champion(a, b):

    # Compare F1-score first
    if a["f1_score"] != b["f1_score"]:
        return a if a["f1_score"] > b["f1_score"] else b

    # Compare Exact Match if F1 is equal
    if a["exact_match"] != b["exact_match"]:
        return a if a["exact_match"] > b["exact_match"] else b

    # Lower latency wins if quality is equal
    return (
        a if a["avg_latency_ms"] <= b["avg_latency_ms"]
        else b
    )


# Select winner
champion = select_champion(model_a, model_b)

# Identify challenger
challenger = (
    model_b if champion["model"] == model_a["model"]
    else model_a
)


# Display comparison
print("\n========== MODEL COMPARISON ==========")

print("MiniLM F1:", model_a["f1_score"])
print("RoBERTa F1:", model_b["f1_score"])

print("MiniLM Latency:", model_a["avg_latency_ms"], "ms")
print("RoBERTa Latency:", model_b["avg_latency_ms"], "ms")


print("\n========== FINAL DECISION ==========")

print("CHAMPION:", champion["model"])
print("CHALLENGER:", challenger["model"])

print("Reason: Better performance based on our selection rule.")


# Prepare decision report
decision = {

    "champion": champion["model"],
    "champion_model_id": champion["model_id"],

    "challenger": challenger["model"],
    "challenger_model_id": challenger["model_id"],

    "selection_rule": [
        "Highest F1-score",
        "Highest Exact Match",
        "Lowest average latency"
    ],

    "evaluation_samples": results["evaluation_samples"],

    "champion_metrics": champion,
    "challenger_metrics": challenger

}


# Save decision report
with open("outputs/champion_decision.json", "w") as file:
    json.dump(decision, file, indent=4)


# Log comparison into MLflow
with mlflow.start_run(run_name="compare_qa"):

    mlflow.log_params({

        "selection_method": "F1_EM_Latency",
        "champion_model": champion["model"],
        "challenger_model": challenger["model"]

    })

    mlflow.log_metrics({

        "champion_f1": champion["f1_score"],
        "challenger_f1": challenger["f1_score"],

        "champion_exact_match": champion["exact_match"],
        "challenger_exact_match": challenger["exact_match"],

        "champion_latency_ms": champion["avg_latency_ms"],
        "challenger_latency_ms": challenger["avg_latency_ms"]

    })

    mlflow.set_tags({

        "decision": "champion_selected",
        "champion": champion["model"],
        "challenger": challenger["model"]

    })

    mlflow.log_artifact(
        "outputs/champion_decision.json"
    )


print("\nDecision saved successfully!")
print("MLflow comparison run recorded!")
