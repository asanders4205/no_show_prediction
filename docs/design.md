### Business Context

### Metric choice
**PR-AUC (Average Precision)** 
* Summarizes the precision-recall curve into a single threshold-independent metric. Unlike ROC-AUC, it is sensitive to class imbalance and is more informative when the positive class is rare or when the business cost of false positives vs. false negatives is asymmetric.
* These models must be sensitive to class imbalance as the vast majority of patients in the dataset as well as in a real clinical setting show up to appointments
* The cost of a false positive is **high**, leading to cancelled appointments, potentially poorer patinet outcomes, poorer patient experience, and potentially higher administrative cost for rescheduling appointments

**Precision**
* Due to high cost of False Postive, the model should optimize for precision. A False Positive in this case is when a patient is predicted to be a no-show and shows up.

* Cost to patient is administrative burden, possible no-show fee at the business's discression, poor experience with the company and/or department.

* Cost to business is reputational harm, administrative costs for rescheduling an appointment, possible increased appointment work queue / backlog.

### Model choice
* Logistic Regression to establish simple binary classification baseline
* Random Forest to get better performance metrics, avoid overfitting, and detect complex feature dynamics
 

### Threshold choice
* Using a hardcoded 50% decision threshold yielded mixed results, the models performed better with threshold sweeping

### False Negative and False Positive cost asymmetry

**False Positive**: Patient was predicted as no-show and showed up/ intented to show up.

* Cost to patient is administrative burden, possible no-show fee at the business's discression, poor experience with the company and/or department.

* Cost to business is reputational harm, administrative costs for rescheduling an appointment, possible increased appointment work queue / backlog.

**False Negative**: Patient did not show up and was predicted as showing up

* Cost to patinet is nothing. Possible no-show fee implemented by the business at their discression

* Cost to business is unutilized appointment slot, increased work queue / backlog if patient reschedules, unaffected work queue / backlog if patient does not reschedule.


### Decisions
* Due to the imbalanced cost of False Positives (predicted no-shows who actually showed up / would have shown up) the model should be optimized for Precision.





