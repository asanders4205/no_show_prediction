# No-Show Prediction

An end-to-end automated ML pipeline on Databricks that predicts whether patients will attend their scheduled medical appointments. The pipeline covers data loading, feature engineering, model training with hyperparameter tuning, MLflow experiment tracking, and automatic promotion of the best model to a production alias.


## What I learned
See 'journal.md' for a full description and my though process during development

## Key decision points:
1. Feature Enginering; Encoding categorical variables

### Problem
When encoding string variables, the `Neighborhood` feature was causing the model to become too large. This is because there are 80+ distinct values for `Neighborhood` in this dataset.

#### Solution: 
I used TargetEncoder for high-cardinality features, i.e. `Neighborhood`

#### Tradeoffs of this approach
TargetEncoder uses the target variable to create a float value for the encoding of the feature. This can cause leakage, so should only be used on the training dataset

#### Result
TBD


2. Class imbalance
#### Problem
The target variable has an 80/20 split; 80% of patients showed up for appointments and 20% did not.

#### Solution:
I added a `weight` column to penalize missed appointments






## Dataset

**Source:** [Kaggle healthcare-no-shows-appointments-dataset](https://www.kaggle.com/datasets/iamtanmayshukla/healthcare-no-shows-appointments-dataset)

**Size:** ~107,000 rows of Brazilian healthcare appointment records  

**Target:** `Showed_up` — 1 if the patient attended, 0 if they did not

| Feature | Type | Description |
|---|---|---|
| `Gender` | Categorical | Patient gender |
| `Age` | Numeric | Patient age |
| `Neighbourhood` | Categorical | Clinic neighbourhood |
| `Scholarship` | Boolean | Enrolled in social welfare program |
| `Hipertension` | Boolean | Has hypertension |
| `Diabetes` | Boolean | Has diabetes |
| `Alcoholism` | Boolean | Has alcoholism |
| `Handicap` | Boolean | Has a handicap |
| `SMS_received` | Boolean | Received an SMS reminder |
| `ScheduledDay` | Date | Date appointment was booked |
| `AppointmentDay` | Date | Date of the appointment |

`PatientId` and `AppointmentID` are dropped before training.

---

<!-- ## Pipeline

```
Load CSV with explicit schema
        ↓
Cast int/bool → double, drop high-null columns, drop ID columns
        ↓
80/20 train/test split (seed=42)
        ↓
CyclicalDateTransformer  →  sin/cos encoding for ScheduledDay & AppointmentDay (12 features)
        ↓
StringIndexer + OneHotEncoder  →  Gender, Neighbourhood
        ↓
VectorAssembler  →  single feature vector
        ↓
GBTClassifier + CrossValidator (3-fold, grid: maxDepth=[3,5], maxIter=[20,50])
        ↓
Evaluate on test set (AUC, Accuracy, F1, Precision, Recall)
        ↓
MLflow: log params, metrics, feature schema, register model as `noshows_gbt`
        ↓
Promote best version to @champion alias
``` -->

### Feature engineering details

- **Cyclical date encoding** — month, day-of-year, and day-of-week are each encoded as a sin/cos pair so the model sees the circular nature of calendar time (e.g. Dec 31 and Jan 1 are adjacent). Applied to both `ScheduledDay` and `AppointmentDay`, yielding 12 derived features.
- **Categorical encoding** — `StringIndexer` followed by `OneHotEncoder` (drop-last) for `Gender` and `Neighbourhood`.
- The `CyclicalDateTransformer` is the first stage in the serialised `Pipeline`, so the logged MLflow model artifact handles raw date columns at inference without any external preprocessing.

<!-- ---

## Model

| Parameter | Value |
|---|---|
| Algorithm | Gradient Boosted Trees (`GBTClassifier`) |
| Hyperparameter search | Grid: `maxDepth` ∈ {3, 5}, `maxIter` ∈ {20, 50} |
| Cross-validation | 3-fold on training set |
| Selection metric | AUC (area under ROC) |
| Random seed | 42 |

--- -->

## MLflow & Model Registry

- **Experiment:** `/Users/asanders4205@gmail.com/predict_no_show`
- **Registered model:** `LogisticRegressionModel`, `RandomForestModel`
- Each run logs params, 5 metrics, and a `feature_schema.json` artifact.
<!-- - The best version from the run is tagged and promoted to the `@champion` alias.

Load the champion model for inference:

```python
import mlflow.spark
model = mlflow.spark.load_model("models:/noshows_gbt@champion")
predictions = model.transform(new_data_df)
```
 -->
---

## Files

```
noshows-prediction/
├── ml_pipeline_agent.py          # Production pipeline — run as a Databricks Job
├── No-show prediction.ipynb      # Exploratory development notebook
├── No-show prediction dev.ipynb  # Development variant with extra validation steps
└── input-datasets/
    └── healthcare_noshows.csv    # Raw dataset
```

---

<!-- ## Running the pipeline

1. Upload the repo to your Databricks workspace.
2. Update `DATASET_PATH` in `ml_pipeline_agent.py` to point to the CSV in your workspace.
3. Run `ml_pipeline_agent.py` as a Databricks Job (or attach it to a cluster and run all cells).
4. The job will train, evaluate, register, and promote the model automatically.

To schedule retraining, create a Databricks Job pointing to `ml_pipeline_agent.py` and set a cron trigger (e.g. weekly). -->




### Work in progress
* Hyperparameter tuning
* More in-depth run logging
* Promotion to `champion` alias


