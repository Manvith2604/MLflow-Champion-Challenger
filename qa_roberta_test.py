from transformers import pipeline

# Load RoBERTa model
qa = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

# Same context used for MiniLM
context = """
MS Ramaiah Institute of Technology was established in 1962.
It is located in Bengaluru, Karnataka.
The institute offers undergraduate and postgraduate courses.
"""

# Same question used for MiniLM
question = "When was MSRIT established?"

# Get prediction
result = qa(
    question=question,
    context=context
)

print("Model: RoBERTa")
print("Question:", question)
print("Answer:", result["answer"])
print("Confidence:", result["score"])