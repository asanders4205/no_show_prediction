# No-Show Prediction

An end-to-end ML pipeline on Databricks that predicts whether patients will attend their scheduled medical appointments. The pipeline covers data loading, feature engineering, model training with hyperparameter tuning, MLflow experiment tracking, and automatic promotion of the best model to a production alias.


## Dataset

**Source:** [Kaggle healthcare-no-shows-appointments-dataset](https://www.kaggle.com/datasets/iamtanmayshukla/healthcare-no-shows-appointments-dataset)

**Size:** ~107,000 rows of Brazilian healthcare appointment records  

**Target:** `Showed_up` — 1 if the patient attended, 0 if they did not

| Feature | Type | Description |
|---|---|---|
| `Gender` | Categorical | Patient gender |
| `Age` | Numeric | Patient age |
| `Neighborhood` | Categorical | Clinic neighbourhood (~81 distinct values) |
| `Scholarship` | Boolean | Enrolled in social welfare program |
| `Hypertension` | Boolean | Has hypertension |
| `Diabetes` | Boolean | Has diabetes |
| `Alcoholism` | Boolean | Has alcoholism |
| `Handicap` | Boolean | Has a handicap |
| `SMS_received` | Boolean | Received an SMS reminder |
| `ScheduledDay` | Date | Date appointment was booked |
| `AppointmentDay` | Date | Date of the appointment |

**Derived feature:** `date_diff` = days between `ScheduledDay` and `AppointmentDay`

`PatientId` and `AppointmentID` are dropped before training.

---

## Pipeline

```
↓
Dataset loaded from folder into bronze Delta table
        ↓
Load CSV into bronze Delta table
        ↓
Data Preprocessing
        * Schema validation with pre-defined expectations
        * Column renaming and casting to match expectations
        * Drop high-null columns, drop ID columns
        * Add unique ID for primary key
        ↓
Feature Engineering
        * Compute date_diff (days between scheduling and appointment)
        * Determine feature importance
        * 80/20 train/test split (seed=42)
        * Class imbalance handling (weightCol ≈ 4:1 for minority class)
        * Cyclical Date Encoding  →  sin/cos encoding for ScheduledDay & AppointmentDay (8 features: month + day-of-year sin/cos per date)
        * Encode Neighborhood as a numeric vector with TargetEncoder (~81 values)
        * Encode Gender similarly with StringIndexer + OneHotEncoder
        * Fit indexer and encoder on models
        * Scale train and test sets with MinMaxScaler
        * Apply class weights to handle imbalance (ratio of Maj to Min classes)
        ↓
Model training
        * Select N most relevant features using Recursive Feature Elimination
        * (LR: 7 features, RF: 12 features)
        ↓
Model training with CrossValidator (3-fold)
        * LR: AUC evaluator, RF: F1 (weighted) evaluator
        * Threshold tuned via F1 sweep post-training
        ↓
Promote best version to @champion alias
```

### Feature engineering details

- **`date_diff`** — Days between scheduling and appointment date. Strong predictor — longer lead time correlates with higher no-show rate. Known at prediction time (no leakage).
- **Cyclical date encoding** — month and day-of-year are each encoded as a sin/cos pair so the model sees the circular nature of calendar time (e.g. Dec 31 and Jan 1 are adjacent). Applied to both `ScheduledDay` and `AppointmentDay`, yielding 8 derived features.
- **Categorical encoding**:
  - `Neighborhood` (~81 values) → **TargetEncoder** (sklearn) — replaces each neighbourhood with its mean no-show rate on the training set. Prevents OHE explosion and captures geographic patterns. Fit on training set only to avoid leakage.
  - `Gender` (2 values) → `StringIndexer` + `OneHotEncoder` (drop-last).
- **Numerical features (passed through as doubles):** `Age`, `Scholarship`, `Hypertension`, `Diabetes`, `Alcoholism`, `Handicap`, `SMS_received`, plus `date_diff` and all 8 cyclical date features.
- Cyclical encoding is performed inline in the feature engineering module (`features/engineering.py`) using Spark column operations (`sin`, `cos`, `month`, `dayofyear`).

### Class imbalance handling

The dataset is ~80% show / ~20% no-show (~4:1 ratio). To prevent the model from always predicting "show" and achieving misleading 80% accuracy:

- **`weightCol`** with weight for no-shows (minority class) derived from the training-set count ratio (show_count / no_show_count). For Logistic Regression this is ~1.98; for Random Forest ~4.0. Shows receive weight 1.0.
- **Primary metrics:** PR-AUC (Average Precision) for model comparison + F1 (weighted) + Recall on no-show class (healthcare sensitivity)
- Accuracy is NOT a primary metric due to imbalance

---

## Model Progression

The pipeline trains two models with the same train/test split, same features, and same `weightCol` for direct comparison:

| Model | Tuning Strategy | Hyperparameters | RFE Features | CV Evaluator | Threshold | Purpose |
|---|---|---|---|---|---|---|
| `LogisticRegression` | CrossValidator 3-fold, parallelism=6 | `elasticNetParam` ∈ {0, 0.5, 1.0}<br>`regParam` ∈ {0.001, 0.01, 0.05, 0.1, 0.5}<br>`maxIter` ∈ {100, 500, 1000} | 7 | AUC | 0.55 | Interpretable baseline |
| `RandomForestClassifier` | CrossValidator 3-fold, parallelism=4 | `numTrees` = 10<br>`maxDepth` = 10<br>`minInstancesPerNode` = 20<br>`featureSubsetStrategy` = sqrt | 12 | F1 (weighted) | 0.30 | Champion model |

**Selection metric:** PR-AUC (Average Precision) — chosen over ROC-AUC for sensitivity to class imbalance  
**Random seed:** 42 for reproducibility

### Results

| Metric | Random Forest (Champion) | Logistic Regression (Baseline) |
|---|---|---|
| PR-AUC | 0.9143 | 0.8837 |
| F1 (weighted) | 0.7496 | 0.7446 |
| Best Threshold | 0.30 | 0.55 |
| No-show Recall (Class 0) | — | 0.3121 |

The Random Forest outperforms the Logistic Regression baseline on both PR-AUC (+0.0306) and weighted F1 (+0.0050). The LR baseline's no-show recall of 0.31 means ~69% of actual no-shows go undetected — the RF model partially addresses this through non-linear feature interactions.

*GBTClassifier is planned as a future model in the progression.*

---

## MLflow & Model Registry

- **Experiment:** `/Users/<your_email>/noshows-pipeline-agent`
- **Registered models:** `noshows_random_forest` (champion) and `noshows_logistic_regression` (baseline)
- Each run logs:
  - **Parameters:** All model hyperparameters and pipeline config
  - **Metrics:** PR-AUC, F1 (weighted), precision, recall, AUC
  - **Artifacts:** `feature_schema.json` for schema validation
  - **Tags:** `run_id`, `dataset`, `validated_by` metadata
- The best version from each training run is promoted to the `@champion` alias.

**Load the champion model for inference:**

```python
import mlflow.spark

# Random Forest (champion)
model = mlflow.spark.load_model("models:/noshows_random_forest@champion")
predictions = model.transform(new_data_df)

# Logistic Regression (baseline)
lr_model = mlflow.spark.load_model("models:/noshows_logistic_regression@champion")
```
---

## Files

```
no_show_prediction/
├── config.yaml                   # Config with table paths and model paths
├── features/
│   └── engineering.py            # Feature engineering module (RFE, encoding, scaling)
├── models/
│   ├── logistic_regression/      # LR training notebook (ID: 1606283588053603)
│   └── random_forest/             # RF training notebook (ID: 1606283588053604)
├── docs/
│   ├── README.md                 # This file
│   ├── design.md                  # Business context, metric choice, cost asymmetry
│   ├── model_card_random_forest.md    # RF model card
│   └── model_card_logistic_regression.md  # LR model card
└── input-datasets/
    └── healthcare_noshows.csv    # Raw dataset (~107K rows)
```

**Dataset path:** `/Workspace/Users/<your_email>/databricks_repo/noshows-prediction/input-datasets/healthcare_noshows.csv`

---

## Running the pipeline

### Prerequisites

**Install dependencies** (only needed for notebooks, not for jobs with proper cluster config):

```python
%pip install -r requirements.txt
```

Required packages:
- `scikit-learn>=1.3.0` (for TargetEncoder)
- `pandas>=2.0.0` (data manipulation)

*Note: PySpark, MLflow, NumPy, and Matplotlib are pre-installed on Databricks.*

### Execution

1. **Model training:** Run the `models/logistic_regression` and `models/random_forest` notebooks in the Databricks notebook UI
   - Each notebook loads data via `config.yaml`, runs feature engineering from `features/engineering.py`, trains with CrossValidator, and logs to MLflow
2. **Feature engineering:** The `features/engineering.py` module is imported by both model notebooks via `config.yaml`'s `features_path` key

**To schedule retraining:** Create a Databricks Job pointing to a model training notebook and set a cron trigger (e.g. weekly).




---

## Project Status

**This is a learning project** focused on teaching end-to-end ML pipeline development on Databricks.

**Completed:**
* ✅ Cyclical date encoding (sin/cos for month + day-of-year on both date columns)
* ✅ TargetEncoder for high-cardinality categorical features (Neighborhood)
* ✅ Class imbalance handling with `weightCol`
* ✅ Recursive Feature Elimination (7 features for LR, 12 for RF)
* ✅ CrossValidator hyperparameter tuning (LR: 45-param grid, RF: sparse grid)
* ✅ Threshold optimization via F1 sweep (LR: 0.55, RF: 0.30)
* ✅ PR-AUC (Average Precision) as primary model comparison metric
* ✅ MLflow experiment tracking and artifact logging
* ✅ Model Registry integration with `@champion` alias promotion
* ✅ Model cards for both Random Forest and Logistic Regression

**Potential enhancements:**
* GBTClassifier as the next model in the progression
* Model interpretation (SHAP values, feature importance)
* Deployment as a Model Serving endpoint
* A/B testing framework for model comparison
* Fairness audit across protected attributes (age, gender, neighbourhood)


