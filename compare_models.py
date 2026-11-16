
import json
import os
import mlflow

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "QA_Champion_Challenger"

PRIMARY_METRIC = "f1_score"
MARGIN = 1.0

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

with open("outputs/metrics_summary.json", "r") as file:
    results = json.load(file)

model_a = results["model_A"]
model_b = results["model_B"]

# --------------------------------------------------
# SELECT CHAMPION
# --------------------------------------------------

def select_champion(a, b):

    f1_delta = a["f1_score"] - b["f1_score"]

    if abs(f1_delta) >= MARGIN:
        return a if f1_delta > 0 else b

    if a["exact_match"] != b["exact_match"]:
        return a if a["exact_match"] > b["exact_match"] else b

    return (
        a if a["avg_latency_ms"] <= b["avg_latency_ms"]
        else b
    )


champion = select_champion(model_a, model_b)

challenger = (
    model_b if champion["model"] == model_a["model"]
    else model_a
)

# Challenger minus Champion, as required by assignment
delta_primary_metric = (
    challenger[PRIMARY_METRIC] - champion[PRIMARY_METRIC]
)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n========== FINAL DECISION ==========")
print("Champion:", champion["model"])
print("Challenger:", challenger["model"])
print("Primary metric:", PRIMARY_METRIC)
print("Minimum margin:", MARGIN)
print("Challenger minus Champion:", delta_primary_metric)

# --------------------------------------------------
# SAVE DECISION REPORT
# --------------------------------------------------

decision = {
    "champion": champion["model"],
    "champion_model_id": champion["model_id"],
    "challenger": challenger["model"],
    "challenger_model_id": challenger["model_id"],
    "primary_metric": PRIMARY_METRIC,
    "metric_direction": "higher_is_better",
    "margin": MARGIN,
    "delta_primary_metric": delta_primary_metric,
    "selection_rule": [
        "F1 score is the primary metric",
        "Minimum improvement margin is 1 percentage point",
        "Exact Match is the first tie-breaker",
        "Lower average latency is the final tie-breaker"
    ],
    "evaluation_samples": results["evaluation_samples"],
    "champion_metrics": champion,
    "challenger_metrics": challenger
}

with open("outputs/champion_decision.json", "w") as file:
    json.dump(decision, file, indent=4)

# --------------------------------------------------
# LOG COMPARISON TO MLFLOW
# --------------------------------------------------

with mlflow.start_run(run_name="compare_qa"):

    mlflow.log_params({
        "primary_metric": PRIMARY_METRIC,
        "margin": MARGIN,
        "metric_direction": "higher_is_better",
        "champion_model": champion["model"],
        "challenger_model": challenger["model"],
        "selection_method": "F1_EM_Latency"
    })

    mlflow.log_metrics({
        "champion_f1": champion["f1_score"],
        "challenger_f1": challenger["f1_score"],
        "champion_exact_match": champion["exact_match"],
        "challenger_exact_match": challenger["exact_match"],
        "champion_latency_ms": champion["avg_latency_ms"],
        "challenger_latency_ms": challenger["avg_latency_ms"],
        "delta_primary_metric": delta_primary_metric
    })

    mlflow.set_tags({
        "winner": champion["model"],
        "challenger": challenger["model"],
        "decision": "champion_selected"
    })

    # Log decision report
    mlflow.log_artifact("outputs/champion_decision.json")

    # Log supporting documentation
    artifacts = [
        ("README.md", "documentation"),
        ("outputs/qualitative_analysis.csv", "qualitative_analysis"),
        ("outputs/metrics_summary.json", "summary"),
        ("data/eval_inputs.csv", "dataset")
    ]

    for path, artifact_folder in artifacts:
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_folder)
            print("Logged:", path)
        else:
            print("Warning: File not found:", path)

print("\nComparison run logged successfully!")
