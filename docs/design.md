### Business Context

Clinics lose billable slots and waste clinical resources when patients miss appointments. Predicting no-shows in advance lets staff proactively overbook or send targeted reminders. The dataset is ~80% show / 20% no-show, so a naive "always predict showed up" classifier achieves 80% accuracy but is clinically useless.

### Metric choice

**PR-AUC (Average Precision)** — primary model comparison metric
* Summarizes the precision-recall curve into a single threshold-independent metric. Unlike ROC-AUC, it is sensitive to class imbalance and is more informative when the positive class is rare or when the business cost of false positives vs. false negatives is asymmetric.
* These models must be sensitive to class imbalance as the vast majority of patients in the dataset as well as in a real clinical setting show up to appointments.

**F1 (weighted)** — CrossValidator evaluator for Random Forest
* Balances precision and recall across both classes, weighted by class frequency. Chosen over AUC for RF because it aligns CV optimization with the business objective (catching no-shows without excessive false alarms).

**AUC (ROC)** — CrossValidator evaluator for Logistic Regression
* Threshold-independent measure of ranking quality. Used for the baseline model to establish discriminative ability before optimizing for the no-show class.

**No-show Recall (Class 0)** — reported as a secondary metric
* Measures what fraction of actual no-shows the model catches. Critical for healthcare sensitivity — missing a no-show means a wasted slot. The LR baseline catches only 31% of actual no-shows.

**Precision** — monitored but not the sole optimization target
* Due to the high cost of false positives (see cost asymmetry below), precision is monitored closely. However, optimizing for precision alone would miss too many actual no-shows. The LR baseline achieves 37% no-show precision.

### Model choice

* **Logistic Regression** — established as a simple, interpretable baseline. Trained with CrossValidator (3-fold, parallelism=6) over a 45-combination grid (elasticNetParam × regParam × maxIter). Uses 7 RFE-selected features. Best threshold: 0.55. Result: PR-AUC = 0.8837, F1 = 0.7446.
* **Random Forest** — selected as the champion model for better performance and ability to capture non-linear feature interactions without overfitting. Trained with CrossValidator (3-fold, parallelism=4) over a sparse grid (10 trees, maxDepth=10, minInstancesPerNode=20, featureSubsetStrategy=sqrt). Uses 12 RFE-selected features. Best threshold: 0.30. Result: PR-AUC = 0.9143, F1 = 0.7496.
* **GBTClassifier** — planned as a future model in the progression for potentially higher performance.

### Class imbalance handling

The dataset is 80% show / 20% no-show (~4:1 majority:minority ratio). To prevent the model from always predicting "show" and achieving misleading 80% accuracy:

* **`weightCol`** — no-shows (minority class) receive a higher weight derived from the training-set count ratio (`show_count / no_show_count`). Shows receive weight 1.0.
* **Logistic Regression** uses `sqrt(show_count / no_show_count) ≈ 1.98` — a milder correction because the raw ratio (~3.94) was too aggressive, pushing accuracy below the majority-class baseline.
* **Random Forest** uses the raw ratio `show_count / no_show_count ≈ 3.94` — tree-based models are more robust to aggressive weighting.
* Accuracy is NOT a primary metric due to imbalance. PR-AUC and F1 (weighted) are used instead.

### Threshold choice

* Using the default 0.50 decision threshold yielded suboptimal results. Threshold sweeping from 0.30 to 0.70 (step 0.05) identified the best weighted F1:
  - **Random Forest:** 0.30 (F1 = 0.7496)
  - **Logistic Regression:** 0.55 (F1 = 0.7446)

### False Negative and False Positive cost asymmetry

**False Positive**: Patient was predicted as no-show and showed up / intended to show up.

* Cost to patient is administrative burden, possible no-show fee at the business's discretion, poor experience with the company and/or department.
* Cost to business is reputational harm, administrative costs for rescheduling an appointment, possible increased appointment work queue / backlog.

**False Negative**: Patient did not show up and was predicted as showing up.

* Cost to patient is nothing. Possible no-show fee implemented by the business at their discretion.
* Cost to business is unutilized appointment slot, increased work queue / backlog if patient reschedules, unaffected work queue / backlog if patient does not reschedule.

### Feature engineering decisions

* **`date_diff`** — days between scheduling and appointment date. The strongest predictor (RF feature importance = 0.594). Longer lead time correlates with higher no-show rate. Known at prediction time (no leakage).
* **Cyclical date encoding** — month and day-of-year are encoded as sin/cos pairs so the model sees the circular nature of calendar time (e.g. Dec 31 and Jan 1 are adjacent). Applied to both `ScheduledDay` and `AppointmentDay`, yielding 8 derived features.
* **Neighborhood** (~81 values) → **TargetEncoder** (sklearn) — replaces each neighbourhood with its mean no-show rate on the training set. Prevents One Hot Encoder explosion and captures geographic patterns. Fit on training set only to avoid leakage.
* **Gender** (2 values) → `StringIndexer` + `OneHotEncoder` (drop-last). Flexible to accomodate more datasets, use-cases, and social contexts.
* **Recursive Feature Elimination (RFE)** — reduces the feature set before training. LR retains 7 features, RF retains 12. Prevents overfitting and speeds up training.
* **MinMaxScaler** — scales all features to [0, 1]. Fit on training set only to prevent leakage.

### Decisions

* PR-AUC is the primary model comparison metric due to class imbalance sensitivity.
* F1 (weighted) is used as the RF CrossValidator evaluator to align CV optimization with the business objective.
* Class imbalance is handled via `weightCol` with model-specific ratios (LR: sqrt ratio ≈ 1.98, RF: raw ratio ≈ 3.94).
* Thresholds are tuned post-training via F1 sweep rather than using the default 0.50.
* RFE is applied to reduce dimensionality and prevent overfitting (LR: 7 features, RF: 12 features).
