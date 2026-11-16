
import mlflow
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")

COMPARE_RUN_ID = "c29294bad632412ba52c2eda406f08dd"
FILE_PATH = "outputs/qualitative_analysis.csv"

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        "Create outputs/qualitative_analysis.csv first."
    )

with mlflow.start_run(run_id=COMPARE_RUN_ID):
    mlflow.log_artifact(
        FILE_PATH,
        artifact_path="qualitative_analysis"
    )

print("Qualitative analysis logged successfully!")
