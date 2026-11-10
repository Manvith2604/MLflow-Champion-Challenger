
import mlflow

# Connect Python to local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create our experiment
experiment_name = "QA_Champion_Challenger"

mlflow.set_experiment(experiment_name)

print("MLflow experiment created successfully!")
print("Experiment name:", experiment_name)
