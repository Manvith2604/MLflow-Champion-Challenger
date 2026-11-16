
import json
import gc
import torch
import mlflow

from mlflow import MlflowClient
from transformers import pipeline

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "QA_Champion_Challenger"

# Both models will be registered under this ONE name
REGISTRY_NAME = "QA_Question_Answering"

DEVICE = 0 if torch.cuda.is_available() else -1

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient(tracking_uri=TRACKING_URI)

# --------------------------------------------------
# READ CHAMPION DECISION
# --------------------------------------------------

with open("outputs/champion_decision.json", "r") as file:
    decision = json.load(file)

champion_name = decision["champion"]
challenger_name = decision["challenger"]

print("Champion:", champion_name)
print("Challenger:", challenger_name)
print("Device:", "CUDA GPU" if DEVICE == 0 else "CPU")

# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

models = [
    {
        "name": "MiniLM",
        "model_id": "deepset/minilm-uncased-squad2"
    },
    {
        "name": "RoBERTa",
        "model_id": "deepset/roberta-base-squad2"
    }
]

# --------------------------------------------------
# REGISTER BOTH MODELS UNDER ONE NAME
# --------------------------------------------------

for model_data in models:

    model_name = model_data["name"]
    model_id = model_data["model_id"]

    if model_name == champion_name:
        alias = "champion"
    elif model_name == challenger_name:
        alias = "challenger"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print("\nRegistering:", model_name)

    qa = None

    try:
        with mlflow.start_run(
            run_name=f"register_{model_name.lower()}"
        ):

            # Log registration details
            mlflow.log_params({
                "model_name": model_name,
                "hf_model_id": model_id,
                "task": "question-answering",
                "role": alias,
                "device": "cuda:0" if DEVICE == 0 else "cpu"
            })

            # Load Hugging Face QA pipeline
            qa = pipeline(
                "question-answering",
                model=model_id,
                device=DEVICE
            )

            # Log model into MLflow
            model_info = mlflow.transformers.log_model(
                transformers_model=qa,
                name=f"qa_pipeline_{model_name.lower()}",
                task="question-answering",
                save_pretrained=True
            )

            run_id = mlflow.active_run().info.run_id

            # Register under the SAME registry name
            registered_model = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=REGISTRY_NAME,
                await_registration_for=300
            )

            version = registered_model.version

            print("Registered model:", REGISTRY_NAME)
            print("Model:", model_name)
            print("Version:", version)

        # Assign champion/challenger alias
        client.set_registered_model_alias(
            name=REGISTRY_NAME,
            alias=alias,
            version=version
        )

        # Add model version metadata
        tags = {
            "model_name": model_name,
            "hf_model_id": model_id,
            "role": alias,
            "student_usn": "1MS25CI073",
            "source_run_id": run_id
        }

        for key, value in tags.items():
            client.set_model_version_tag(
                name=REGISTRY_NAME,
                version=version,
                key=key,
                value=value
            )

        print("Alias assigned:", alias)

    finally:
        del qa
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --------------------------------------------------
# VERIFY REGISTRATION
# --------------------------------------------------

print("\n========== REGISTRY VERIFICATION ==========")

registered = client.get_registered_model(REGISTRY_NAME)

for version in registered.latest_versions:
    print(
        "Version:", version.version,
        "| Run ID:", version.run_id,
        "| Model:", version.tags.get("model_name"),
        "| Role:", version.tags.get("role")
    )

for alias in ["champion", "challenger"]:
    version = client.get_model_version_by_alias(
        REGISTRY_NAME,
        alias
    )

    print(
        f"\n{alias.upper()} -> Version {version.version}"
    )

print("\nMODEL REGISTRATION COMPLETED!")
