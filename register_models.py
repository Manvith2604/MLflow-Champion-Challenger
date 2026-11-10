
import json
import mlflow

from mlflow import MlflowClient
from transformers import pipeline


# -----------------------------------------
# 1. CONNECT TO MLFLOW
# -----------------------------------------

tracking_uri = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("QA_Champion_Challenger")

client = MlflowClient(tracking_uri=tracking_uri)


# -----------------------------------------
# 2. READ CHAMPION DECISION
# -----------------------------------------

with open("outputs/champion_decision.json", "r") as file:
    decision = json.load(file)

champion_name = decision["champion"]


# -----------------------------------------
# 3. DEFINE MODELS
# -----------------------------------------

models = [

    {
        "name": "MiniLM",
        "model_id": "deepset/minilm-uncased-squad2",
        "registry_name": "QA_MiniLM"
    },

    {
        "name": "RoBERTa",
        "model_id": "deepset/roberta-base-squad2",
        "registry_name": "QA_RoBERTa"
    }

]


# -----------------------------------------
# 4. REGISTER BOTH MODELS
# -----------------------------------------

for model_data in models:

    model_name = model_data["name"]
    model_id = model_data["model_id"]
    registry_name = model_data["registry_name"]

    # Determine alias automatically
    alias = (
        "champion"
        if model_name == champion_name
        else "challenger"
    )

    print("\nRegistering:", model_name)

    with mlflow.start_run(
        run_name=f"register_{model_name.lower()}"
    ):

        # Load Hugging Face pipeline
        qa = pipeline(
            "question-answering",
            model=model_id,
            device=-1
        )

        # Log model into MLflow
        model_info = mlflow.transformers.log_model(
            transformers_model=qa,
            name="qa_pipeline",
            task="question-answering",
            save_pretrained=True
        )

        # Register the logged model
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=registry_name,
            await_registration_for=300
        )

        version = registered_model.version

        print("Registered model:", registry_name)
        print("Version:", version)

    # Assign alias to registered version
    client.set_registered_model_alias(
        name=registry_name,
        alias=alias,
        version=version
    )

    # Add metadata to model version
    client.set_model_version_tag(
        name=registry_name,
        version=version,
        key="role",
        value=alias
    )

    print("Alias assigned:", alias)

    del qa


print("\n================================")
print("MODEL REGISTRATION COMPLETED")
print("================================")
