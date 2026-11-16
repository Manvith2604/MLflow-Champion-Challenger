
import mlflow
from mlflow.tracking import MlflowClient
import os

TRACKING_URI = "http://127.0.0.1:5000"

RUN_ID = "c29294bad632412ba52c2eda406f08dd"
README_PATH = "README.md"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

if not os.path.exists(README_PATH):
    raise FileNotFoundError("README.md not found in project folder.")

client.log_artifact(
    run_id=RUN_ID,
    local_path=README_PATH,
    artifact_path="documentation"
)

print("README logged successfully!")
print("MLflow Run ID:", RUN_ID)
print("Artifact path: documentation/README.md")
