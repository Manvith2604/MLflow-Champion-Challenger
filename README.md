# Champion–Challenger Model Comparison Using Hugging Face and MLflow

## 1. Project Overview

This project implements a **Champion–Challenger Model Comparison System** using Hugging Face Transformers and MLflow.

The objective is to evaluate two Question Answering (QA) models on the same evaluation dataset, compare their performance using quantitative metrics, automatically select the better-performing model, and manage both models through the MLflow Model Registry.

The project follows an end-to-end MLOps workflow that includes:

- Dataset preparation and validation.
- Hugging Face model evaluation.
- Performance and latency measurement.
- Experiment tracking using MLflow.
- Automated Champion–Challenger selection.
- Model registration and alias management.
- Loading the Champion model directly from MLflow.
- Testing the registered model on new, unseen examples.

The project uses two pretrained extractive Question Answering models and evaluates them using the same 40-question dataset.

## 2. Project Objectives

The main objectives are:

1. Evaluate two pretrained Hugging Face Question Answering models using identical inputs.
2. Measure Exact Match (EM), F1 Score, average inference latency, and total inference time.
3. Track model parameters, metrics, artifacts, and experiment runs using MLflow.
4. Implement an automated Champion–Challenger selection strategy.
5. Register both models under a common MLflow registered model name.
6. Assign `champion` and `challenger` aliases based on evaluation results.
7. Load the Champion model through its MLflow alias without directly loading the Hugging Face model in the prediction script.
8. Generate predictions on new examples and save the results as a CSV file.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Hugging Face Transformers | Loading and executing pretrained QA models |
| PyTorch | Deep learning inference |
| MLflow | Experiment tracking and model management |
| Pandas | Dataset handling and result processing |
| NumPy | Numerical operations |
| Scikit-learn | Supporting machine learning utilities |
| Matplotlib | Evaluation visualizations |
| Git and GitHub | Version control and project hosting |
| CUDA | GPU-accelerated inference when available |

## 4. Models Used

Two pretrained Question Answering models from Hugging Face are compared.

### Model A – MiniLM

- Model name: `deepset/minilm-uncased-squad2`
- Architecture: MiniLM
- Task: Extractive Question Answering
- Dataset family: SQuAD 2.0

### Model B – RoBERTa

- Model name: `deepset/roberta-base-squad2`
- Architecture: RoBERTa
- Task: Extractive Question Answering
- Dataset family: SQuAD 2.0

Both models receive the same context and question pairs in the same order. This ensures that the comparison is fair and consistent.

## 5. System Architecture

The project follows this workflow:

```text
             Evaluation Dataset
              eval_inputs.csv
                     |
                     v
              Dataset Validation
                     |
          +----------+----------+
          |                     |
          v                     v
     MiniLM Model          RoBERTa Model
          |                     |
          v                     v
     Predictions            Predictions
          |                     |
          v                     v
     EM, F1, Latency      EM, F1, Latency
          |                     |
          +----------+----------+
                     |
                     v
              MLflow Tracking
                     |
                     v
          Champion–Challenger
               Comparison
                     |
                     v
             Champion Selection
                     |
                     v
              MLflow Registry
                     |
          +----------+----------+
          |                     |
          v                     v
       Champion              Challenger
       Alias                    Alias
          |                     |
          +----------+----------+
                     |
                     v
             Champion Prediction
                     |
                     v
            New Test Dataset
                new_data.csv
                     |
                     v
             Prediction Results
          new_data_predictions.csv
```

## 6. Dataset Description

The project uses two CSV datasets.

### 6.1 Evaluation Dataset

File: `data/eval_inputs.csv`

The evaluation dataset contains 40 Question Answering examples.

Each example includes:

| Column | Description |
|---|---|
| id | Unique identifier for the example |
| context | Passage containing the information |
| question | Question asked about the passage |
| reference_answer | Expected answer used for evaluation |

The dataset contains both answerable and unanswerable questions. This allows the models to be evaluated on their ability to answer questions and abstain when the required information is unavailable.

Both models use this exact dataset for evaluation.

### 6.2 New Test Dataset

File: `data/new_data.csv`

This dataset contains five new test examples that are separate from the evaluation dataset.

It is used to test the registered Champion model after model selection.

The prediction script reads the examples from this file and saves the generated answers and confidence scores.

## 7. Evaluation Metrics

The models are evaluated using the following metrics.

### 7.1 Exact Match (EM)

Exact Match measures whether the predicted answer exactly matches the reference answer after text normalization.

\[
EM =
\frac{\text{Number of exact matches}}
{\text{Total examples}}\times 100
\]

A higher Exact Match score indicates better exact-answer performance.

### 7.2 F1 Score

F1 Score measures the overlap between the predicted answer and the reference answer at the token level.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

Where:

- Precision measures how many predicted tokens are correct.
- Recall measures how many reference tokens are captured.

A higher F1 Score indicates better answer overlap.

### 7.3 Average Inference Latency

Average inference latency measures the average time required to generate a prediction.

\[
Average\ Latency =
\frac{Total\ Inference\ Time}{Number\ of\ Examples}
\]

It is reported in milliseconds.

### 7.4 Total Inference Time

Total inference time represents the combined time taken by the model to process the complete evaluation dataset.

## 8. Champion–Challenger Selection Strategy

The project uses an automated selection strategy to determine the Champion model.

The selection rules are:

1. F1 Score is the primary evaluation metric.
2. A minimum improvement margin of 1 percentage point is required.
3. If the primary metric does not establish a winner, Exact Match is used as the first tie-breaker.
4. If the tie remains, the model with lower average inference latency is preferred.

The comparison process generates:

`outputs/champion_decision.json`

This file records:

- Selected Champion model.
- Challenger model.
- Model identifiers.
- Primary metric.
- Improvement margin.
- Difference in primary metric.
- Evaluation sample count.
- Metrics for both models.
- Selection rules.

The decision is also logged as an MLflow experiment run.

## 9. Experimental Results

Both models were evaluated using the same 40 examples.

### 9.1 Performance Comparison

| Evaluation Metric | RoBERTa | MiniLM |
|---|---:|---:|
| Evaluation samples | 40 | 40 |
| Exact Match | 87.50% | 85.00% |
| F1 Score | 89.17% | 86.67% |
| Average latency | 13.44 ms | 16.59 ms |
| Total inference time | 0.5375 s | 0.6635 s |

### 9.2 Result Analysis

RoBERTa achieved an F1 Score of 89.17%, while MiniLM achieved 86.67%.

The difference in F1 Score is:

\[
89.17 - 86.67 = 2.50
\]

The improvement is 2.50 percentage points, which exceeds the required improvement margin of 1 percentage point.

RoBERTa also achieved:

- Higher Exact Match.
- Higher F1 Score.
- Lower average inference latency.
- Lower total inference time.

Therefore, RoBERTa was selected as the Champion model.

### 9.3 Final Model Selection

| Role | Model | Hugging Face Model ID |
|---|---|---|
| Champion | RoBERTa | `deepset/roberta-base-squad2` |
| Challenger | MiniLM | `deepset/minilm-uncased-squad2` |

**Final Champion: RoBERTa**

## 10. MLflow Experiment Tracking

MLflow is used to track and organize the experiments.

Experiment name:

`QA_Champion_Challenger`

The evaluation process creates separate runs for both models and a comparison run.

### Model Evaluation Runs

Each model run records:

**Parameters**
- Hugging Face model ID.
- Task name.
- Model role.
- Evaluation sample count.
- Random seed.
- Device used.
- Inference settings.

**Metrics**
- Exact Match.
- F1 Score.
- Average inference latency.
- Total inference time.

**Artifacts**
- Prediction CSV.
- Evaluation dataset.
- Metrics summary JSON.
- Accuracy comparison plot.
- Latency comparison plot.
- Confidence distribution plot.

### Comparison Run

The comparison run records the Champion selection decision and its supporting information.

The following artifact is logged:

`champion_decision.json`

The qualitative analysis CSV is also logged to the comparison run.

## 11. MLflow Model Registry

Both models are registered under a single registered model name:

`QA_Question_Answering`

The registry contains multiple model versions.

### Final Registry Configuration

| Model | Version | Alias |
|---|---:|---|
| RoBERTa | 4 | champion |
| MiniLM | 3 | challenger |

The aliases provide a convenient way to reference models without hardcoding their version numbers.

The prediction script loads the Champion using:

```python
MODEL_URI = "models:/QA_Question_Answering@champion"
```

If a new model becomes the Champion in a future evaluation, its alias can be updated in the registry without changing the prediction script.

## 12. Prediction Using the Registered Champion

The file `predict_from_mlflow.py` loads the registered Champion model directly from MLflow.

It does not directly import or call the Hugging Face `pipeline()` function.

The script performs the following steps:

1. Connects to the MLflow tracking server.
2. Retrieves the Champion model version using its alias.
3. Reads the Hugging Face model ID from model metadata.
4. Loads the model using the MLflow Transformers flavor.
5. Reads new questions and contexts from `data/new_data.csv`.
6. Generates predictions and confidence scores.
7. Saves the results to a CSV file.

Model loading:

```python
model = mlflow.transformers.load_model(
    "models:/QA_Question_Answering@champion",
    device=DEVICE
)
```

Output file:

`outputs/new_data_predictions.csv`

The output contains:

- Example ID.
- Question.
- Predicted answer.
- Confidence score.
- Model version.
- Hugging Face model ID.

The final prediction test successfully loaded Champion Version 4 and generated five predictions.

## 13. Qualitative Analysis

Qualitative analysis is performed to understand differences between the two models beyond numerical metrics.

The observed prediction differences are stored in:

`outputs/qualitative_analysis.csv`

Four differing outputs were identified in the 40-example evaluation:

| Example ID | Observation |
|---|---|
| 7 | Both models give the correct answer with slightly different wording. |
| 8 | Both models correctly identify the location; wording differs. |
| 29 | Both models fail to calculate the date difference correctly. |
| 34 | RoBERTa abstains because the manufacturer is not specified, while MiniLM produces an unsupported answer. |

This analysis highlights differences in answer generation and abstention behavior.

The qualitative analysis is logged as an artifact in the MLflow comparison run.

## 14. Project Folder Structure

```text
ML_FLOW_PROJECT/
│
├── data/
│   ├── eval_inputs.csv
│   └── new_data.csv
│
├── outputs/
│   ├── accuracy_comparison.png
│   ├── champion_decision.json
│   ├── confidence_distribution.png
│   ├── latency_comparison.png
│   ├── metrics_summary.json
│   ├── new_data_predictions.csv
│   ├── predictions_champion_candidate.csv
│   ├── predictions_challenger_candidate.csv
│   └── qualitative_analysis.csv
│
├── screenshots/
│
├── check_dataset.py
├── compare_models.py
├── evaluate.py
├── log_qualitative.py
├── mlflow_setup.py
├── predict_from_mlflow.py
├── qa_roberta_test.py
├── qa_test.py
├── register_models.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

Generated local MLflow tracking directories and the Python virtual environment are excluded from version control.

## 15. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Manvith2604/MLflow-Champion-Challenger.git
```

Navigate into the project directory:

```bash
cd MLflow-Champion-Challenger
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows Git Bash:

```bash
source venv/Scripts/activate
```

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU inference, install a PyTorch build compatible with the system's NVIDIA GPU and CUDA environment.

### Step 4: Start the MLflow Tracking Server

Open a separate terminal with the project environment activated.

Run:

```bash
python -m mlflow server --host 127.0.0.1 --port 5000
```

Open the MLflow UI:

http://127.0.0.1:5000

Keep the tracking server running while executing the project scripts.

## 16. Execution Instructions

Execute the following scripts from the project root directory.

### Step 1: Evaluate Both Models

```bash
python evaluate.py
```

This evaluates both models on the shared evaluation dataset and logs the results to MLflow.

### Step 2: Compare the Models

```bash
python compare_models.py
```

This compares the evaluation metrics and selects the Champion model.

### Step 3: Register the Models

```bash
python register_models.py
```

This registers both models under `QA_Question_Answering` and assigns the appropriate aliases.

### Step 4: Log Qualitative Analysis

```bash
python log_qualitative.py
```

This logs the qualitative analysis CSV to the comparison run.

### Step 5: Test the Registered Champion

```bash
python predict_from_mlflow.py
```

This loads the Champion through its MLflow alias and predicts answers for the new test examples.

## 17. Expected Outputs

After successful execution, the project generates:

| Output | Description |
|---|---|
| `predictions_champion_candidate.csv` | Predictions from one evaluated model |
| `predictions_challenger_candidate.csv` | Predictions from the other evaluated model |
| `metrics_summary.json` | Evaluation metrics for both models |
| `champion_decision.json` | Final Champion selection details |
| `accuracy_comparison.png` | Accuracy comparison visualization |
| `latency_comparison.png` | Inference latency comparison |
| `confidence_distribution.png` | Prediction confidence visualization |
| `qualitative_analysis.csv` | Analysis of observed prediction differences |
| `new_data_predictions.csv` | Predictions generated using the registered Champion |

The MLflow UI provides access to experiment runs, model artifacts, metrics, registered model versions, and aliases.

## 18. GitHub Repository

Project repository:

https://github.com/Manvith2604/MLflow-Champion-Challenger

The repository contains the source code, datasets, output files, requirements, documentation, and project screenshots.

The project uses Git for version control and maintains multiple commits documenting the development process.

## 19. Conclusion

This project demonstrates an end-to-end Champion–Challenger workflow using Hugging Face and MLflow.

Two pretrained Question Answering models were evaluated on the same dataset using Exact Match, F1 Score, and inference latency.

Based on the evaluation results, RoBERTa achieved an F1 Score of 89.17%, outperforming MiniLM, which achieved 86.67%.

RoBERTa was therefore selected as the Champion and registered as Version 4 under the `champion` alias. MiniLM was registered as Version 3 under the `challenger` alias.

The final prediction script successfully loaded the Champion directly from the MLflow Model Registry and generated predictions on five new examples.

The project demonstrates experiment tracking, model evaluation, automated model selection, model versioning, alias management, and reproducible inference as part of an MLOps workflow.

---

**Project:** Champion–Challenger Model Comparison Using Hugging Face and MLflow  
**Task:** Question Answering  
**Evaluation Dataset:** 40 examples  
**Champion:** RoBERTa  
**Challenger:** MiniLM  
**Experiment:** QA_Champion_Challenger  
**Registered Model:** QA_Question_Answering
