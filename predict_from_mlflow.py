
import mlflow

# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the registered Champion model
model_uri = "models:/QA_MiniLM@champion"

print("Loading Champion model...")

qa = mlflow.transformers.load_model(model_uri)

print("Champion model loaded successfully!\n")

# New test examples
test_cases = [
    {
        "context": "The capital of France is Paris.",
        "question": "What is the capital of France?"
    },
    {
        "context": "Python was created by Guido van Rossum in 1991.",
        "question": "Who created Python?"
    },
    {
        "context": "The Earth revolves around the Sun once every 365 days.",
        "question": "How long does Earth take to revolve around the Sun?"
    },
    {
        "context": "The Pacific Ocean is the largest ocean on Earth.",
        "question": "Which is the largest ocean?"
    },
    {
        "context": "The human heart has four chambers.",
        "question": "How many chambers does the heart have?"
    }
]

# Generate answers
for i, item in enumerate(test_cases, start=1):

    result = qa(
        question=item["question"],
        context=item["context"]
    )

    print(f"Test {i}")
    print("Question:", item["question"])
    print("Predicted Answer:", result["answer"])
    print("Confidence:", round(result["score"], 4))
    print("-" * 45)
