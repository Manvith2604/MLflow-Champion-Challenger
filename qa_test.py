from transformers import pipeline

# Load the pretrained model
qa = pipeline(
    "question-answering",
    model="deepset/minilm-uncased-squad2"
)

# Context
context = """
MS Ramaiah Institute of Technology was established in 1962.
It is located in Bengaluru, Karnataka.
The institute offers undergraduate and postgraduate courses.
"""

# Question
question = "Who is the principal of MSRIT"

# Get answer
result = qa(
    question=question,
    context=context
)

print("Question:", question)
print("Answer:", result["answer"])
print("Confidence:", result["score"])