<!---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
--->

# Model Card for No-Show Prediction (Logistic Regression)

A Logistic Regression binary classifier that predicts whether a patient will attend or miss their medical appointment. Trained as the baseline model on ~107K appointment records using PySpark ML on Databricks, with cyclical date encoding, target-encoded neighbourhoods, and class-weight balancing for the ~4:1 show/no-show imbalance. Serves as a performance reference for the Random Forest champion model.

## Model Details

### Model Description

This model predicts patient no-shows for medical appointments using a Logistic Regression classifier. It was developed as the baseline model in a model progression (LogisticRegression -> RandomForestClassifier -> GBTClassifier) to establish a simple, interpretable benchmark before exploring more complex models. It uses the same "Medical Appointment No Shows" dataset from Kaggle (110,527 appointments from the Brazilian public health system) and the same feature engineering pipeline as the Random Forest model.

A key difference: Recursive Feature Elimination (RFE) selects **7 features** for the Logistic Regression model (vs. 12 for Random Forest), reflecting the fact that linear models benefit from more aggressive dimensionality reduction to avoid overfitting on irrelevant features.

Class imbalance is addressed via a `weightCol` where no-show samples (class 0) receive a weight of 1.9842 and show samples (class 1) receive 1.0. The decision threshold was tuned via an F1 sweep to 0.55, higher than the Random Forest's 0.30, reflecting the different probability distributions produced by the linear model.

- **Developed by:** Alec Sanders
- **Funded by [optional]:** N/A — personal learning project
- **Shared by [optional]:** N/A
- **Model type:** Supervised binary classification (Logistic Regression)
- **Language(s) (NLP):** N/A — tabular data, no NLP
- **License:** MIT
- **Finetuned from model [optional]:** N/A — trained from scratch

### Model Sources [optional]

- **Repository:** See `config.yaml` in the project root (`no_show_prediction/`)
- **Paper [optional]:** N/A
- **Demo [optional]:** N/A

## Uses

### Direct Use

Predicting whether a patient will fail to attend a scheduled medical appointment. As the baseline model, it is primarily used for performance comparison against the Random Forest champion model. It can also serve as a lightweight, interpretable fallback when model transparency is prioritised over predictive power.

### Downstream Use [optional]

The model could be embedded in a clinic scheduling system as an interpretable first-pass filter, flagging appointments at risk of no-show for review by a more complex model or human staff. Its linear coefficients provide direct feature-level interpretability that tree ensembles do not offer natively.

### Out-of-Scope Use

- **Individual clinical decisions:** This model predicts attendance behaviour, not medical outcomes. It should not be used to deny care, deprioritise patients clinically, or make individual treatment decisions.
- **Non-medical appointment settings:** The model was trained exclusively on Brazilian public-health appointment data. It should not be applied to other healthcare systems, geographies, or appointment types without retraining and validation.
- **Demographic profiling:** The model should not be used to infer general patient reliability or to discriminate against specific neighbourhoods or socioeconomic groups.
- **Primary production model:** The Random Forest model (PR-AUC 0.9143) outperforms this baseline (PR-AUC 0.8837) and should be preferred for production use unless interpretability is the overriding concern.

## Bias, Risks, and Limitations

- **Geographic bias:** The training data originates from a single Brazilian public health system. Appointment attendance patterns are influenced by local infrastructure, culture, and economics. Results may not generalise to other regions or healthcare systems.
- **Socioeconomic bias:** The `Scholarship` feature (Bolsa Familia welfare enrolment) and `Neighborhood` target encoding encode socioeconomic information. The model may learn patterns that disproportionately flag patients from lower-income areas as no-show risks.
- **Class imbalance:** The dataset is ~80% show / ~20% no-show. While `weightCol` mitigates this during training, the model's no-show recall is only 0.3121 — meaning ~69% of actual no-shows go undetected at the tuned threshold. This is the primary performance limitation.
- **False positive cost:** A false positive (patient predicted as no-show who would have attended) carries a high cost: the appointment slot may be given away, the patient may face administrative burden or no-show fees, and the clinic may suffer reputational harm.
- **Linearity assumption:** Logistic Regression assumes a linear relationship between features and the log-odds of the target. Real-world no-show behaviour likely involves non-linear interactions (e.g., age x SMS_received) that this model cannot capture.
- **Data vintage:** The dataset is a static snapshot. Seasonal trends, policy changes, or post-pandemic behaviour shifts are not captured.
- **No protected-attribute audit:** The model has not been formally audited for fairness across age, gender, or neighbourhood subgroups.

### Recommendations

Users (both direct and downstream) should be made aware that:

- The model provides probabilistic risk scores, not deterministic predictions. Human judgement should remain in the loop for any action taken on a flagged appointment.
- This is a baseline model. The Random Forest champion (`noshows_random_forest@champion`) achieves higher PR-AUC and F1 and should be preferred for production use.
- The no-show recall of 0.31 means the model misses the majority of actual no-shows. Threshold tuning should be revisited if higher sensitivity is required.
- Regular retraining is recommended to capture seasonal and behavioural drift.
- A fairness audit across protected attributes (age, gender, neighbourhood) should be conducted before any deployment.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import mlflow

# Load the champion model from MLflow Model Registry
model = mlflow.spark.load_model("models:/noshows_logistic_regression@champion")

# Predict on new appointment data (must match the training feature schema)
predictions = model.transform(new_appointments_df)

# The model outputs a 'probability' column (vector of [P(no-show), P(show)])
# and a 'prediction' column (0 = no-show, 1 = showed up).
# A tuned threshold of 0.55 was used during development — adjust as needed.
```

## Training Details

### Training Data

The training data is the "Medical Appointment No Shows" dataset from Kaggle, containing 110,527 appointment records. After an explicit `StructType` schema load and column renaming (`Neighbourhood` -> `Neighborhood`, `Hipertension` -> `Hypertension`), the data was split 80/20 (seed=42) into:

- **Training set:** 85,397 rows (26 columns including weightCol and scaledFeatures vector)
- **Test set:** 21,590 rows (25 columns)

Raw columns: `PatientId`, `AppointmentID` (dropped), `Gender`, `ScheduledDay`, `AppointmentDay`, `Age`, `Neighborhood`, `Scholarship`, `Hypertension`, `Diabetes`, `Alcoholism`, `Handicap`, `SMS_received`, `Showed_up` (target).

Source file: See `dataset_path` in `config.yaml`

### Training Procedure

#### Preprocessing [optional]

The preprocessing pipeline is identical to the Random Forest model (same training notebook, same feature engineering module). The only difference is the final RFE selection count: 7 features for Logistic Regression vs. 12 for Random Forest.

1. **Schema enforcement:** Explicit `StructType` on CSV load to prevent schema drift.
2. **Column renaming:** `Neighbourhood` -> `Neighborhood`, `Hipertension` -> `Hypertension`.
3. **Type casting:** `Showed_up` (bool -> double) as `labelCol`. All feature columns cast to appropriate numeric types.
4. **Null audit:** Columns with >60% nulls flagged and dropped.
5. **Cyclical date encoding:** `ScheduledDay` and `AppointmentDay` transformed into 8 sin/cos features (month, day-of-year for each). Raw date columns dropped.
6. **Neighbourhood target encoding:** sklearn `TargetEncoder` (binary, smooth="auto") fit on the training set only to prevent leakage. Encoded value = mean no-show rate per neighbourhood.
7. **Gender encoding:** `StringIndexer` -> `OneHotEncoder` (2 values, OHE appropriate).
8. **Vector assembly:** All features assembled into a `features` vector column.
9. **Min-Max scaling:** `MinMaxScaler` fit on training data only, applied to both splits.
10. **Feature selection:** Recursive Feature Elimination (RFE) with sklearn `LogisticRegression` as the selector, reducing 18 features to **7 selected features**.
11. **Class weighting:** `weightCol` = 1.9842 for no-shows (class 0), 1.0 for shows (class 1). Ratio derived from training-set counts.

RFE-selected features (7): a subset of the 12 features selected for Random Forest, chosen to minimise multicollinearity and overfitting in the linear model. The VectorSlicer extracts these from `scaledFeatures` into `selectedFeatures`.

#### Training Hyperparameters

- **Training regime:** fp32, CPU-only (serverless Spark)
- **Model:** `pyspark.ml.classification.LogisticRegression`
- **weightCol:** `weightCol` (1.9842 for no-show class, 1.0 for show class)
- **CrossValidator:** 3-fold, parallelism=6
- **Evaluator:** AUC (area under ROC)
- **ParamGrid:**
  - `elasticNetParam`: [0.0, 0.5, 1.0] (L2 -> mixed -> L1 regularization)
  - `regParam`: [0.001, 0.01, 0.05, 0.1, 0.5]
  - `maxIter`: [100, 500, 1000]
  - Total combinations: 3 x 5 x 3 = 45 parameter sets
- **Threshold:** 0.55 (tuned via F1 sweep post-training)
- **seed:** 42

#### Speeds, Sizes, Times [optional]

- **Compute:** Databricks Serverless CPU (Spark Connect)
- **Training time:** <3 minutes (3-fold CV with 45-param grid on 85K rows — Logistic Regression converges faster than Random Forest)
- **Feature engineering:** ~2 minutes (shared with RF pipeline — target encoding + scaling + RFE)
- **Model size:** Very lightweight (single linear model, coefficient vector only)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Held-out 20% test split (21,590 rows) from the original 110,527 records. No overlap with training data (seed=42 random split). Identical test set used for both Logistic Regression and Random Forest evaluation.

#### Factors

Evaluation was conducted on the overall test set. No subgroup disaggregation by age, gender, neighbourhood, or health condition was performed during development.

#### Metrics

- **PR-AUC (Average Precision):** Summarises the precision-recall curve into a single threshold-independent metric. Chosen over ROC-AUC because it is sensitive to class imbalance and the positive class (no-show) is the minority (~20%). This is the primary model comparison metric.
- **F1 Score (weighted):** Harmonic mean of precision and recall, weighted by class support.
- **Precision:** Emphasised because false positives (patient predicted as no-show who actually attends) carry high operational and reputational cost.
- **Recall (no-show class):** Tracks sensitivity — the proportion of actual no-shows the model successfully identifies. This is the key weakness of the baseline model.

### Results

| Metric | Logistic Regression (Baseline) | Random Forest (Champion) |
| --- | --- | --- |
| PR-AUC | 0.8837 | 0.9143 |
| F1 (weighted) | 0.7446 | 0.7496 |
| Best Threshold | 0.55 | 0.30 |
| No-show Recall (Class 0) | 0.3121 | — |

The Logistic Regression baseline achieves solid PR-AUC (0.8837) and weighted F1 (0.7446), but trails the Random Forest champion on both metrics (PR-AUC -0.0306, F1 -0.0050). The no-show recall of 0.3121 is the most significant gap — the model identifies only ~31% of actual no-shows at the tuned threshold of 0.55. This reflects the linear model's inability to capture non-linear feature interactions (e.g., the combined effect of long lead time and lack of SMS reminder).

#### Summary

The Logistic Regression baseline establishes a reasonable performance benchmark (PR-AUC 0.8837, F1 0.7446) but is limited by its linearity assumption and low no-show recall (0.31). It is suitable as an interpretable reference model but should not be used as the primary production model. The Random Forest champion is recommended for deployment.

## Model Examination [optional]

Logistic Regression provides native interpretability through its coefficient vector. Each coefficient represents the change in log-odds of no-show per unit increase in the corresponding (scaled) feature. Key observations:

- **`date_diff`:** Expected to have a positive coefficient (longer lead time -> higher no-show probability).
- **`SMS_received`:** Expected to have a negative coefficient (SMS reminder -> lower no-show probability).
- **`Neighborhood_te`:** Coefficient reflects the linear contribution of the target-encoded no-show rate per neighbourhood.

The 7-feature model (vs. 12 for RF) means some cyclical date features were eliminated by RFE, suggesting the linear model found them redundant or non-informative on their own.

No formal SHAP or LIME analysis was conducted. The coefficient-based interpretation is itself a key advantage of this model over the Random Forest.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Databricks Serverless CPU (Spark Connect)
- **Hours used:** <0.3 hours total (feature engineering + training + evaluation)
- **Cloud Provider:** AWS
- **Compute Region:** N/A (serverless, region depends on workspace)
- **Carbon Emitted:** Negligible — single linear model, serverless auto-scales to zero when idle.

## Technical Specifications [optional]

### Model Architecture and Objective

- **Algorithm:** `pyspark.ml.classification.LogisticRegression`
- **Objective:** Binary classification — minimise regularised negative log-likelihood (logistic loss) with elastic net regularization.
- **Input:** Vector of 7 RFE-selected, min-max scaled features (extracted from `scaledFeatures` via VectorSlicer into `selectedFeatures`).
- **Output:** `prediction` (0 = no-show, 1 = showed up), `probability` (vector of [P(no-show), P(show)]), `rawPrediction`.
- **Class weighting:** `weightCol` 1.9842 for minority (no-show, class 0), 1.0 for majority (show, class 1).
- **Threshold:** 0.55 (applied post-training via F1 sweep).
- **Regularization:** Elastic net (L1 + L2), tuned via CrossValidator grid.

### Compute Infrastructure

Databricks Serverless CPU compute (Spark Connect). No dedicated cluster required — auto-provisioned for notebook execution and auto-scales to zero when idle.

#### Hardware

AWS-backed serverless CPU instances. No GPU required for this linear model.

#### Software

- **Databricks Runtime:** Serverless (latest)
- **PySpark ML:** `pyspark.ml.classification.LogisticRegression`, `CrossValidator`, `ParamGridBuilder`
- **scikit-learn:** `TargetEncoder`, `RFE`, `LogisticRegression` (feature selection only)
- **MLflow:** Experiment tracking, model registry, champion alias promotion
- **Experiment path:** See `mlflow.experiment_path` in `config.yaml`
- **Registered model:** `noshows_logistic_regression` with `@champion` alias
- **Training notebook:** `models/logistic_regression` (notebook ID: 1606283588053603)

## Glossary [optional]

- **No-show:** A patient who did not attend their scheduled appointment (`Showed_up = 0`).
- **Showed up:** A patient who attended their scheduled appointment (`Showed_up = 1`).
- **PR-AUC (Average Precision):** Area under the Precision-Recall curve. More informative than ROC-AUC for imbalanced datasets because it focuses on the minority (positive) class.
- **date_diff:** Number of days between when the appointment was scheduled and the appointment date itself. Longer lead times correlate with higher no-show rates.
- **Target encoding:** Replacing a categorical value (e.g., neighbourhood) with the mean of the target variable (no-show rate) for that category, computed from the training set only.
- **Cyclical date encoding:** Representing calendar periods (month, day-of-year) as sine/cosine pairs so that the model understands the cyclic nature of time (e.g., December is close to January).
- **weightCol:** A per-row weight column passed to the classifier to counteract class imbalance. No-show rows receive a higher weight (1.9842) so the model pays more attention to the minority class.
- **RFE (Recursive Feature Elimination):** An iterative feature selection method that recursively removes the least important features and refits until the desired number of features is reached.
- **Elastic net regularization:** A combination of L1 (Lasso) and L2 (Ridge) regularization, controlled by `elasticNetParam` (0 = pure L2, 1 = pure L1).
- **Champion alias:** An MLflow Model Registry alias pointing to the best-performing version of a registered model, used for production inference.

## More Information [optional]

This model was developed as a learning project to build an end-to-end ML pipeline on Databricks. It serves as the first model in a planned progression: LogisticRegression (baseline) -> RandomForestClassifier (champion) -> GBTClassifier (future). Each model uses the same train/test split, same feature engineering pipeline, and same MLflow experiment for direct comparison.

The full pipeline includes data loading with explicit schema, type casting, null auditing, cyclical date feature engineering, target encoding, min-max scaling, recursive feature elimination, class-weight balancing, cross-validated hyperparameter tuning, threshold tuning, MLflow logging, and model registry promotion.

## Model Card Authors [optional]

Alec Sanders