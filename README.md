# No-Show Prediction

An end-to-end automated ML pipeline on Databricks that predicts whether patients will attend their scheduled medical appointments. The pipeline covers data loading, feature engineering, model training with hyperparameter tuning, MLflow experiment tracking, and automatic promotion of the best model to a production alias.


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
Load CSV with explicit schema
        ↓
Cast int/bool → double, drop high-null columns, drop ID columns
        ↓
Compute date_diff (days between scheduling and appointment)
        ↓
80/20 train/test split (seed=42)
        ↓
Class imbalance handling (weightCol ≈ 4:1 for minority class)
        ↓
CyclicalDateTransformer  →  sin/cos encoding for ScheduledDay & AppointmentDay (12 features)
        ↓
TargetEncoder  →  Neighborhood (~81 values)
StringIndexer + OneHotEncoder  →  Gender
        ↓
VectorAssembler  →  single feature vector
        ↓
Model training with CrossValidator (3-fold, AUC metric)
        ↓
Promote best version to @champion alias
```

### Feature engineering details

- **`date_diff`** — Days between scheduling and appointment date. Strong predictor — longer lead time correlates with higher no-show rate. Known at prediction time (no leakage).
- **Cyclical date encoding** — month, day-of-year, and day-of-week are each encoded as a sin/cos pair so the model sees the circular nature of calendar time (e.g. Dec 31 and Jan 1 are adjacent). Applied to both `ScheduledDay` and `AppointmentDay`, yielding 12 derived features.
- **Categorical encoding**:
  - `Neighborhood` (~81 values) → **TargetEncoder** (sklearn) — replaces each neighbourhood with its mean no-show rate on the training set. Prevents OHE explosion and captures geographic patterns. Fit on training set only to avoid leakage.
  - `Gender` (2 values) → `StringIndexer` + `OneHotEncoder` (drop-last).
- **Numerical features (passed through as doubles):** `Age`, `Scholarship`, `Hypertension`, `Diabetes`, `Alcoholism`, `Handicap`, `SMS_received`, plus `date_diff` and all 12 cyclical date features.
- The `CyclicalDateTransformer` is the first stage in the serialised `Pipeline`, so the logged MLflow model artifact handles raw date columns at inference without any external preprocessing.

### Class imbalance handling

The dataset is ~80% show / ~20% no-show (~4:1 ratio). To prevent the model from always predicting "show" and achieving misleading 80% accuracy:

- **`weightCol`** with weight ≈ 4.0 for no-shows (minority class), 1.0 for shows
- **Primary metrics:** AUC (model comparison) + Recall on no-show class (healthcare sensitivity)
- Accuracy is NOT a primary metric due to imbalance

---

## Model Progression

The pipeline trains three models with the same train/test split, same features, and same `weightCol` for direct comparison:

| Model | Tuning Strategy | Hyperparameters | Purpose |
|---|---|---|---|
| `LogisticRegression` | None — default params | — | Establish baseline quickly |
| `RandomForestClassifier` | CrossValidator 3-fold | `numTrees` ∈ {50, 100}<br>`maxDepth` ∈ {5, 10} | Ensemble baseline |
| `GBTClassifier` | CrossValidator 3-fold | `maxDepth` ∈ {3, 5}<br>`maxIter` ∈ {20, 50} | Final production model |

**Selection metric:** AUC (area under ROC)  
**Random seed:** 42 for reproducibility

---

## MLflow & Model Registry

- **Experiment:** `/Users/asanders4205@gmail.com/noshows-pipeline-agent`
- **Registered model:** `noshows_gbt`
- Each run logs:
  - **Parameters:** All model hyperparameters and pipeline config
  - **Metrics:** AUC, accuracy, precision, recall, F1 score
  - **Artifacts:** `feature_schema.json` for schema validation
  - **Tags:** `run_id`, `dataset`, `validated_by` metadata
- The best version from each training run is promoted to the `@champion` alias.

**Load the champion model for inference:**

```python
import mlflow.spark
model = mlflow.spark.load_model("models:/noshows_gbt@champion")
predictions = model.transform(new_data_df)
```
---

## Files

```
no_show_prediction/
├── ml_pipeline_agent.py          # Production pipeline — run as a Databricks Job
├── requirements.txt              # Python dependencies (scikit-learn, pandas)
├── data_preprocessing.ipynb      # Data exploration and preprocessing notebook
├── random_forest.ipynb           # RandomForest model training notebook
├── README.md                     # This file
└── input-datasets/
    └── healthcare_noshows.csv    # Raw dataset (~107K rows)
```

**Dataset path:** `/Workspace/Users/asanders4205@gmail.com/databricks_repo/noshows-prediction/input-datasets/healthcare_noshows.csv`

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

1. **Production:** Run `ml_pipeline_agent.py` as a Databricks Job
   - Attach to a cluster (or configure cluster init script with `requirements.txt`)
   - The job trains, evaluates, registers, and promotes the model automatically
2. **Development/EDA:** Use `data_preprocessing.ipynb` and `random_forest.ipynb` in the Databricks notebook UI
   - Run `%pip install -r requirements.txt` in the first code cell

**To schedule retraining:** Create a Databricks Job pointing to `ml_pipeline_agent.py` and set a cron trigger (e.g. weekly).




---

## Project Status

**This is a learning project** focused on teaching end-to-end ML pipeline development on Databricks.

**Completed:**
* ✅ Custom CyclicalDateTransformer for date encoding
* ✅ TargetEncoder for high-cardinality categorical features
* ✅ Class imbalance handling with `weightCol`
* ✅ Hyperparameter tuning with CrossValidator (RandomForest, GBT)
* ✅ MLflow experiment tracking and artifact logging
* ✅ Model Registry integration with `@champion` alias promotion
* ✅ Feature schema validation

**Potential enhancements:**
* Model interpretation (SHAP values, feature importance)
* Threshold optimization for precision/recall tradeoff
* Deployment as a Model Serving endpoint
* A/B testing framework for model comparison


