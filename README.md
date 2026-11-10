
# Champion-Challenger Model Comparison using Hugging Face and MLflow

## 1. Project Overview

This project compares two pretrained Question Answering models using Hugging Face and MLflow.

The objective is to evaluate both models using the same dataset, compare their performance, select a Champion model, and register both models in the MLflow Model Registry.

## 2. Models Used

| Model | Hugging Face Model ID |
|---|---|
| MiniLM | deepset/minilm-uncased-squad2 |
| RoBERTa | deepset/roberta-base-squad2 |

## 3. Dataset

A custom evaluation dataset containing 10 Question Answering examples was used.

Each example contains:
- Context
- Question
- Reference Answer

Both models were evaluated using the same dataset.

## 4. Evaluation Metrics

The following metrics were used:

- Exact Match (EM): Measures whether the predicted answer exactly matches the reference answer.
- F1 Score: Measures the overlap between predicted and reference answers.
- Average Inference Latency: Measures the average time taken to generate an answer.

## 5. Evaluation Results

| Metric | MiniLM | RoBERTa |
|---|---:|---:|
| Exact Match | 100% | 100% |
| F1 Score | 100% | 100% |
| Average Latency | 31.28 ms | 62.12 ms |

Both models achieved the same accuracy on the evaluation dataset. MiniLM was approximately 2 times faster.

## 6. Champion-Challenger Selection

MiniLM was selected as the Champion because it achieved the same EM and F1 scores as RoBERTa with lower inference latency.

- Champion: MiniLM
- Challenger: RoBERTa

## 7. MLflow Integration

MLflow was used for:
- Experiment tracking
- Logging model parameters and metrics
- Storing evaluation artifacts
- Comparing model runs
- Registering both models
- Assigning Champion and Challenger aliases

Registered models:

- QA_MiniLM — champion
- QA_RoBERTa — challenger

## 8. Champion Model Testing

The registered Champion model was loaded directly from the MLflow Model Registry using its alias.

It was tested on five new Question Answering examples to verify that the registered model could generate predictions.

## 9. Technologies Used

- Python
- Hugging Face Transformers
- PyTorch
- MLflow
- Pandas
- Scikit-learn

## 10. Conclusion

The experiment demonstrated how two pretrained Question Answering models can be evaluated and compared using a common dataset.

Both models achieved 100% Exact Match and F1 scores on the 10-example evaluation dataset. MiniLM was selected as the Champion due to its lower inference latency.

The results are specific to this small evaluation dataset and do not establish that MiniLM will outperform RoBERTa on all Question Answering tasks.
