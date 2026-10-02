<!---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for No-Show Prediction (Random Forest)

A Random Forest binary classifier that predicts whether a patient will attend or miss their medical appointment. Trained on ~107K appointment records using PySpark ML on Databricks, with cyclical date encoding, target-encoded neighbourhoods, and class-weight balancing for the ~4:1 show/no-show imbalance.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model predicts patient no-shows for medical appointments. It uses a Random Forest classifier trained on a dataset of 110,527 appointments from the Brazilian public health system (Kaggle "Medical Appointment No Shows"). Features include patient demographics (Age, Gender), health conditions (Hypertension, Diabetes, Alcoholism, Handicap), welfare enrolment (Scholarship), SMS reminders, days between scheduling and appointment (date_diff), and cyclical encodings of the scheduling and appointment dates. Neighbourhood (~81 distinct values) is target-encoded using the training-set no-show rate. Recursive Feature Elimination (RFE) selects the top 12 features from 18 candidates.

Class imbalance is addressed via a `weightCol` where no-show samples receive a weight of ~4.0 (the majority-to-minority ratio) and show samples receive 1.0. The decision threshold was tuned via an F1 sweep rather than using the default 0.5 cutoff.

A Logistic Regression baseline (PR-AUC 0.8837, F1 0.7446) was also trained on the same features for comparison.

- **Developed by:** <your_email>
- **Funded by [optional]:** N/A — personal learning project
- **Shared by [optional]:** N/A
- **Model type:** Supervised binary classification (Random Forest)
- **Language(s) (NLP):** N/A — tabular data, no NLP
- **License:** MIT
- **Finetuned from model [optional]:** N/A — trained from scratch

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** `/Workspace/Users/<your_email>/no_show_prediction/`
- **Paper [optional]:** N/A
- **Demo [optional]:** N/A

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Predicting whether a patient will fail to attend a scheduled medical appointment. The primary use case is enabling clinics to proactively identify high-risk no-show slots and offer them to other patients (e.g., via waitlist or double-booking) to reduce unutilised capacity.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The model could be embedded in a clinic scheduling system to flag appointments at risk of no-show, triggering automated SMS reminders, follow-up calls, or waitlist offers. It could also be used to analyse no-show patterns across neighbourhoods or demographic segments for operational planning.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- **Individual clinical decisions:** This model predicts attendance behaviour, not medical outcomes. It should not be used to deny care, deprioritise patients clinically, or make individual treatment decisions.
- **Non-medical appointment settings:** The model was trained exclusively on Brazilian public-health appointment data. It should not be applied to other healthcare systems, geographies, or appointment types without retraining and validation.
- **Demographic profiling:** The model should not be used to infer general patient reliability or to discriminate against specific neighbourhoods or socioeconomic groups.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- **Geographic bias:** The training data originates from a single Brazilian public health system. Appointment attendance patterns are influenced by local infrastructure, culture, and economics. Results may not generalise to other regions or healthcare systems.
- **Socioeconomic bias:** The `Scholarship` feature (Bolsa Familia welfare enrolment) and `Neighborhood` target encoding encode socioeconomic information. The model may learn patterns that disproportionately flag patients from lower-income areas as no-show risks.
- **Class imbalance:** The dataset is ~80% show / ~20% no-show. While `weightCol` mitigates this during training, the model may still underperform on the no-show class in absolute terms (no-show recall is a known limitation — see Results).
- **False positive cost:** A false positive (patient predicted as no-show who would have attended) carries a high cost: the appointment slot may be given away, the patient may face administrative burden or no-show fees, and the clinic may suffer reputational harm. The model is tuned with a threshold of 0.30 to optimise for F1 and precision, reducing but not eliminating this risk.
- **Data vintage:** The dataset is a static snapshot. Seasonal trends, policy changes, or post-pandemic behaviour shifts are not captured.
- **No protected-attribute audit:** The model has not been formally audited for fairness across age, gender, or neighbourhood subgroups.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware that:

- The model provides probabilistic risk scores, not deterministic predictions. Human judgement should remain in the loop for any action taken on a flagged appointment.
- Threshold tuning should be revisited when deploying to a new clinic or population, as the cost ratio of false positives to false negatives may differ.
- Regular retraining is recommended to capture seasonal and behavioural drift.
- A fairness audit across protected attributes (age, gender, neighbourhood) should be conducted before clinical deployment.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import mlflow

# Load the champion model from MLflow Model Registry
model = mlflow.spark.load_model("models:/noshows_random_forest@champion")

# Predict on new appointment data (must match the training feature schema)
predictions = model.transform(new_appointments_df)

# The model outputs a 'probability' column (vector of [P(no-show), P(show)])
# and a 'prediction' column (0 = no-show, 1 = showed up).
# A tuned threshold of 0.30 was used during development — adjust as needed.
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The training data is the "Medical Appointment No Shows" dataset from Kaggle, containing 110,527 appointment records. After an explicit `StructType` schema load and column renaming (`Neighbourhood` → `Neighborhood`, `Hipertension` → `Hypertension`), the data was split 80/20 (seed=42) into:

- **Training set:** 85,397 rows (26 columns including weightCol and scaledFeatures vector)
- **Test set:** 21,590 rows (25 columns)

Raw columns: `PatientId`, `AppointmentID` (dropped), `Gender`, `ScheduledDay`, `AppointmentDay`, `Age`, `Neighborhood`, `Scholarship`, `Hypertension`, `Diabetes`, `Alcoholism`, `Handicap`, `SMS_received`, `Showed_up` (target).

Source file: `/Workspace/Users/<your_email>/databricks_repo/noshows-prediction/input-datasets/healthcare_noshows.csv`

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

1. **Schema enforcement:** Explicit `StructType` on CSV load to prevent schema drift.
2. **Column renaming:** `Neighbourhood` → `Neighborhood`, `Hipertension` → `Hypertension`.
3. **Type casting:** `Showed_up` (bool → double) as `labelCol`. All feature columns cast to appropriate numeric types.
4. **Null audit:** Columns with >60% nulls flagged and dropped.
5. **Cyclical date encoding:** `ScheduledDay` and `AppointmentDay` transformed into 8 sin/cos features (month, day-of-year for each). Raw date columns dropped.
6. **Neighbourhood target encoding:** sklearn `TargetEncoder` (binary, smooth="auto") fit on the training set only to prevent leakage. Encoded value = mean no-show rate per neighbourhood.
7. **Gender encoding:** `StringIndexer` → `OneHotEncoder` (2 values, OHE appropriate).
8. **Vector assembly:** All features assembled into a `features` vector column.
9. **Min-Max scaling:** `MinMaxScaler` fit on training data only, applied to both splits.
10. **Feature selection:** Recursive Feature Elimination (RFE) with sklearn `LogisticRegression` as the selector, reducing 18 features to 12 selected features.
11. **Class weighting:** `weightCol` = ~4.0 for no-shows (minority), 1.0 for shows (majority). Ratio derived from training-set counts.

RFE-selected features (12): `Age`, `Scholarship`, `SMS_received`, `date_diff`, `Sched_month_cos`, `Sched_dayofyear_sin`, `Sched_dayofyear_cos`, `Appoi_month_sin`, `Appoi_month_cos`, `Appoi_dayofyear_sin`, `Appoi_dayofyear_cos`, `Neighborhood_te`


#### Training Hyperparameters

- **Training regime:** fp32, CPU-only (serverless Spark) <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

- **Compute:** Databricks Serverless CPU (Spark Connect)
- **Training time:** <5 minutes (3-fold CV with 10-tree RF on 85K rows)
- **Feature engineering:** ~2 minutes (target encoding + scaling + RFE)
- **Model size:** Lightweight (~10 trees, depth ≤10)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

Held-out 20% test split (21,590 rows) from the original 110,527 records. No overlap with training data (seed=42 random split).

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Evaluation was conducted on the overall test set. No subgroup disaggregation by age, gender, neighbourhood, or health condition was performed during development.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- **PR-AUC (Average Precision):** Summarises the precision-recall curve into a single threshold-independent metric. Chosen over ROC-AUC because it is sensitive to class imbalance and the positive class (no-show) is the minority (~20%). This is the primary model comparison metric.
- **F1 Score (weighted):** Harmonic mean of precision and recall, weighted by class support. Used as the CrossValidator evaluation metric.
- **Precision:** Emphasised because false positives (patient predicted as no-show who actually attends) carry high operational and reputational cost.
- **Recall (no-show class):** Tracks sensitivity — the proportion of actual no-shows the model successfully identifies.

### Results

| Metric | Random Forest (Champion) | Logistic Regression (Baseline) |
| --- | --- | --- |
| PR-AUC | 0.9143 | 0.8837 |
| F1 (weighted) | 0.7496 | 0.7446 |
| Best Threshold | 0.30 | 0.55 |
| No-show Recall (Class 0) | — | 0.3121 |

The Random Forest model outperforms the Logistic Regression baseline on both PR-AUC (+0.0306) and weighted F1 (+0.0050). The Logistic Regression baseline's no-show recall of 0.31 indicates that ~69% of actual no-shows go undetected at the tuned threshold — the Random Forest model partially addresses this through its ability to capture non-linear feature interactions.

#### Summary

The Random Forest champion model achieves strong PR-AUC (0.9143) and weighted F1 (0.7496) on the held-out test set. The tuned threshold of 0.30 optimises for the F1/precision trade-off appropriate to the asymmetric cost structure (false positives are more costly than false negatives). The model is suitable for pilot deployment with human-in-the-loop oversight.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

Feature importance was examined during development using the Random Forest's built-in `featureImportances` output. The strongest predictors are expected to be:

- **`date_diff`** (days between scheduling and appointment): longer lead time correlates with higher no-show rate — known to be a strong signal.
- **`Age`**: certain age groups show different attendance patterns.
- **`SMS_received`**: patients who received SMS reminders show lower no-show rates.
- **`Neighborhood_te`**: target-encoded neighbourhood captures geographic/socioeconomic attendance patterns.

No formal SHAP or LIME analysis was conducted. This is a candidate for future interpretability work.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Databricks Serverless CPU (Spark Connect)
- **Hours used:** <0.5 hours total (feature engineering + training + evaluation)
- **Cloud Provider:** AWS
- **Compute Region:** N/A (serverless, region depends on workspace)
- **Carbon Emitted:** Negligible — small-scale tree model, serverless auto-scales to zero when idle.

## Technical Specifications [optional]

### Model Architecture and Objective

- **Algorithm:** `pyspark.ml.classification.RandomForestClassifier`
- **Objective:** Binary classification — minimise weighted impurity (Gini) across ensemble of decision trees.
- **Input:** Vector of 12 RFE-selected, min-max scaled features.
- **Output:** `prediction` (0 = no-show, 1 = showed up), `probability` (vector of [P(no-show), P(show)]), `rawPrediction`.
- **Class weighting:** `weightCol` ~4.0 for minority (no-show), 1.0 for majority (show).
- **Threshold:** 0.30 (applied post-training via F1 sweep, not a native RF hyperparameter).

### Compute Infrastructure

Databricks Serverless CPU compute (Spark Connect). No dedicated cluster required — auto-provisioned for notebook execution and auto-scales to zero when idle.

#### Hardware

AWS-backed serverless CPU instances. No GPU required for this tree-based model.

#### Software

- **Databricks Runtime:** Serverless (latest)
- **PySpark ML:** `pyspark.ml.classification.RandomForestClassifier`, `CrossValidator`, `ParamGridBuilder`
- **scikit-learn:** `TargetEncoder`, `RFE`, `LogisticRegression` (feature selection / baseline)
- **MLflow:** Experiment tracking, model registry, champion alias promotion
- **Experiment path:** `/Users/<your_email>/noshows-pipeline-agent`
- **Registered model:** `noshows_random_forest` with `@champion` alias
- **Training hyperparameters:** numTrees=10, maxDepth=10, minInstancesPerNode=20, featureSubsetStrategy=sqrt, CrossValidator 3-fold, parallelism=4, F1 evaluator

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@misc{noshow_prediction_2026,
  title={Patient No-Show Prediction using PySpark ML and MLflow on Databricks},
  author={asanders4205},
  year={2026},
  note={Personal learning project — Random Forest binary classifier for medical appointment no-show prediction}
}
```

**APA:**

asanders4205. (2026). Patient No-Show Prediction using PySpark ML and MLflow on Databricks. Personal learning project.

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

- **No-show:** A patient who did not attend their scheduled appointment (`Showed_up = 0`).
- **Showed up:** A patient who attended their scheduled appointment (`Showed_up = 1`).
- **PR-AUC (Average Precision):** Area under the Precision-Recall curve. More informative than ROC-AUC for imbalanced datasets because it focuses on the minority (positive) class.
- **date_diff:** Number of days between when the appointment was scheduled and the appointment date itself. Longer lead times correlate with higher no-show rates.
- **Target encoding:** Replacing a categorical value (e.g., neighbourhood) with the mean of the target variable (no-show rate) for that category, computed from the training set only.
- **Cyclical date encoding:** Representing calendar periods (month, day-of-year) as sine/cosine pairs so that the model understands the cyclic nature of time (e.g., December is close to January).
- **weightCol:** A per-row weight column passed to the classifier to counteract class imbalance. No-show rows receive a higher weight so the model pays more attention to the minority class.
- **RFE (Recursive Feature Elimination):** An iterative feature selection method that recursively removes the least important features and refits until the desired number of features is reached.
- **Champion alias:** An MLflow Model Registry alias pointing to the best-performing version of a registered model, used for production inference.

## More Information [optional]

This model was developed as a learning project to build an end-to-end ML pipeline on Databricks. The full pipeline includes data loading with explicit schema, type casting, null auditing, cyclical date feature engineering, target encoding, min-max scaling, recursive feature elimination, class-weight balancing, cross-validated training, threshold tuning, MLflow logging, and model registry promotion.

The project also includes a Logistic Regression baseline model (`noshows_logistic_regression`) for comparison, trained on 7 RFE-selected features (vs. 12 for the Random Forest).

## Model Card Authors [optional]

<your_email>

## Model Card Contact

<your_email>