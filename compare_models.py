
import json
import mlflow

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "QA_Champion_Challenger"

PRIMARY_METRIC = "f1_score"
IMPROVEMENT_MARGIN = 1.0  # 1 percentage point

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --------------------------------------------------
# LOAD EVALUATION RESULTS
# --------------------------------------------------

with open("outputs/metrics_summary.json", "r") as file:
    results = json.load(file)

model_a = results["model_A"]
model_b = results["model_B"]


# --------------------------------------------------
# CHAMPION SELECTION
# --------------------------------------------------

def select_champion(a, b):

    f1_delta = a["f1_score"] - b["f1_score"]

    # Primary metric: F1 score
    if abs(f1_delta) >= IMPROVEMENT_MARGIN:
        return a if f1_delta > 0 else b

    # Tie-breaker 1: Exact Match
    if a["exact_match"] != b["exact_match"]:
        return a if a["exact_match"] > b["exact_match"] else b

    # Tie-breaker 2: Lower latency
    return (
        a if a["avg_latency_ms"] <= b["avg_latency_ms"]
        else b
    )


champion = select_champion(model_a, model_b)

challenger = (
    model_b if champion["model"] == model_a["model"]
    else model_a
)

# Positive value means Champion has a higher F1 score
delta_primary_metric = (
    champion["f1_score"] - challenger["f1_score"]
)

# --------------------------------------------------
# DISPLAY DECISION
# --------------------------------------------------

print("\n========== MODEL COMPARISON ==========")

for model in [model_a, model_b]:
    print(f"\n{model['model']}")
    print("F1 Score:", model["f1_score"])
    print("Exact Match:", model["exact_match"])
    print("Average Latency:", model["avg_latency_ms"], "ms")

print("\n========== FINAL DECISION ==========")

print("CHAMPION:", champion["model"])
print("CHALLENGER:", challenger["model"])
print("Primary Metric:", PRIMARY_METRIC)
print("Minimum Improvement Margin:", IMPROVEMENT_MARGIN)
print("F1 Difference:", delta_primary_metric)

# --------------------------------------------------
# CREATE DECISION REPORT
# --------------------------------------------------

decision = {
    "champion": champion["model"],
    "champion_model_id": champion["model_id"],

    "challenger": challenger["model"],
    "challenger_model_id": challenger["model_id"],

    "primary_metric": PRIMARY_METRIC,
    "metric_direction": "higher_is_better",
    "improvement_margin": IMPROVEMENT_MARGIN,
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
        "metric_direction": "higher_is_better",
        "improvement_margin": IMPROVEMENT_MARGIN,
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

    mlflow.log_artifact("outputs/champion_decision.json")

print("\nDecision saved successfully!")
print("MLflow comparison run recorded!")
