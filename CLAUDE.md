# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**This is a learning project.** The goal is to teach the user how to build an end-to-end ML pipeline on Databricks — not to ship production code as fast as possible.

The domain is patient no-show prediction for medical appointments, using PySpark ML and MLflow on Databricks. The primary artifact is `ml_pipeline_agent.py`.

**When assisting in this repo, always pause at key ML decision points and explain the concept + options before writing any code.** The user makes the call; Claude handles the boilerplate. Key decision points include:
- Class imbalance strategy
- Feature selection and encoding choices
- Model family selection
- Hyperparameter tuning strategy
- Which evaluation metrics to prioritise
- Whether and how to interpret the model

## Running the Pipeline

This is a **Databricks-native project** — there are no local CLI build/test commands. Execution happens via:

- **Production:** Run `ml_pipeline_agent.py` as a Databricks Job (attach to cluster, run as script)
- **Development/EDA:** `No-show prediction.ipynb` and `No-show prediction dev.ipynb` in Databricks notebook UI

The dataset path is hardcoded to the Databricks workspace:
```
/Workspace/Users/asanders4205@gmail.com/databricks_repo/noshows-prediction/input-datasets/healthcare_noshows.csv
```

MLflow experiment: `/Users/asanders4205@gmail.com/noshows-pipeline-agent`

## Dependency Management

**`requirements.txt`** contains all Python dependencies not pre-installed on Databricks:

```
scikit-learn>=1.3.0  # For TargetEncoder
pandas>=2.0.0        # Data manipulation
```

**Installation:**
- **Notebooks:** Run `%pip install -r requirements.txt` in the first code cell (already added to `No-show prediction.ipynb`)
- **Jobs/Scripts:** Add `%pip install -r requirements.txt` at the top of `ml_pipeline_agent.py`, or configure as a cluster init script

**Note:** PySpark, MLflow, NumPy, and Matplotlib are pre-installed on Databricks and should NOT be added to `requirements.txt`.

## Pipeline Architecture

### End-to-End Flow

```
CSV Load → Type Casting & Null Audit → Train/Test Split (80/20, seed=42)
  → CyclicalDateTransformer → StringIndexer/OHE → VectorAssembler
  → GBTClassifier with CrossValidator (3-fold, AUC)
  → MLflow logging → Model Registry → @champion alias promotion
```

### Key Files

| File | Purpose |
|---|---|
| `ml_pipeline_agent.py` | Production pipeline — all stages from load to registry |
| `No-show prediction.ipynb` | Full EDA and exploratory training workflow |
| `No-show prediction dev.ipynb` | Development variant with extra validation |
| `requirements.txt` | Python dependencies (scikit-learn, pandas) |
| `input-datasets/healthcare_noshows.csv` | Raw data (~107K rows) |

### Feature Engineering

**CyclicalDateTransformer** (custom Spark Transformer in `ml_pipeline_agent.py`) is the key innovation — it encodes `ScheduledDay` and `AppointmentDay` as 12 cyclical features (sin/cos for month, day-of-year, day-of-week per date column), then drops the raw date columns. It runs first in the pipeline so the serialized model handles dates natively at inference time.

**Categorical:**
- `Gender` → StringIndexer + OneHotEncoder (2 values, OHE is fine)
- `Neighborhood` → TargetEncoder (sklearn) — replaces each neighbourhood with its mean no-show rate on the training set. ~81 distinct values make OHE impractical. Must be fit on `train_df` only to avoid leakage.

**Numerical (passed through as doubles):** `Age`, `Scholarship`, `Hypertension`, `Diabetes`, `Alcoholism`, `Handicap`, `SMS_received`, `date_diff` + all 12 cyclical date features

**`date_diff`:** Days between scheduling and appointment date. Strong predictor — longer lead time correlates with higher no-show rate. Known at prediction time (no leakage).

**Column renames (applied in notebook, must match in `ml_pipeline_agent.py`):**
- `Neighbourhood` → `Neighborhood`
- `Hipertension` → `Hypertension`

**Dropped before training:** `PatientId`, `AppointmentID`

**Target:** `Showed_up` (bool → double, `labelCol`)

### Model

**Planned progression:** LogisticRegression (baseline) → RandomForestClassifier → GBTClassifier

Each model is trained with the same train/test split, same features, same `weightCol`, and logged to the same MLflow experiment for direct comparison.

**Primary metrics:** AUC (model comparison) + Recall on no-show class (healthcare sensitivity). Accuracy is not a primary metric due to class imbalance.

**Class imbalance:** `weightCol` with weight ≈ 4.0 for no-shows (minority), 1.0 for shows. Ratio derived from training set counts.

**Tuning strategy per model:**
| Model | Tuning | Params |
|---|---|---|
| `LogisticRegression` | None — default params, establish baseline quickly | — |
| `RandomForestClassifier` | CrossValidator 3-fold, 2×2 grid | `numTrees ∈ {50, 100}`, `maxDepth ∈ {5, 10}` |
| `GBTClassifier` | CrossValidator 3-fold, 2×2 grid | `maxDepth ∈ {3, 5}`, `maxIter ∈ {20, 50}` |

### MLflow Integration

- Model registered as `noshows_gbt` in the workspace model registry
- Best run tagged with `run_id`, `dataset`, `validated_by` metadata
- Promoted to `@champion` alias after each successful training run
- Load for inference: `mlflow.spark.load_model("models:/noshows_gbt@champion")`
- Feature schema artifact (`feature_schema.json`) logged each run for schema validation

## Problem Framing

- **Task:** Binary classification — predict whether a patient will attend (`Showed_up=True`) or miss (`Showed_up=False`) their appointment
- **Business goal:** Catching a predicted no-show in advance lets the clinic fill the slot with another patient
- **Leakage:** No columns are post-event except the target itself — the feature set is safe

### Class Imbalance

The dataset is ~80% show / ~20% no-show (~4:1 majority:minority ratio). The current pipeline does **not** address this — it trains on raw imbalanced data.

**Known gap:** The pipeline should add `weightCol` (weight ≈ 4.0 for no-shows, 1.0 for shows) and treat AUC + Recall as the primary metrics rather than accuracy. Accuracy is misleading here — a model that always predicts "showed up" is 80% accurate but useless.

Recommended fix (to be applied to `ml_pipeline_agent.py`):
```python
majority = train_df.filter(col(TARGET) == 1.0).count()
minority = train_df.filter(col(TARGET) == 0.0).count()
ratio    = majority / minority

train_df = train_df.withColumn(
    "class_weight",
    when(col(TARGET) == 0.0, ratio).otherwise(1.0)
)
# Then pass weightCol="class_weight" to GBTClassifier
```

## Development Notes

- No traditional test suite — validation is via MLflow metrics logged during training
- Random seed 42 used throughout for reproducibility
- The explicit `StructType` schema on CSV load (in `ml_pipeline_agent.py`) is intentional — avoids schema inference drift across runs
- Columns with >60% nulls are audited and dropped automatically during data prep
