### Journal
#### Tuesday 6/9
  - Problem framing: Binary classification predicting Showed_up, no post-event leakage
  - Class imbalance: weightCol (~4× for no-shows) + AUC and Recall as primary metrics — motivated by the healthcare context where missing a no-show is costly
  - Feature engineering: Neighborhood encoded with TargetEncoder (not OHE — ~81 categories), date_diff added as a feature, column renames (Neighbourhood→Neighborhood,
  Hipertension→Hypertension) noted
  - Model progression: LogisticRegression (no tuning) → RandomForestClassifier (2×2 CV) → GBTClassifier (2×2 CV)
  - Tuning strategy: No tuning on LR baseline; CrossValidator 3-fold 2×2 grid on RF and GBT

  Still open: Gate 6 (evaluation metrics) — the recommendation was AUC + no-show Recall as headline numbers, logging all five metrics, but you hadn't confirmed that before we wrapped up.



## 6/16 
Found out how to connect VectorAssembler with the LinearRegression() function

Also found out target variable needs to be of double datatype




# 6/19
Have to remember to eval the model on the test set, not the train set. Looking forward to seeing these metrics and starting to assess tradeoffs.


### 6/20
LR Model runs and produces metrics. Next step is to restructure the project. Then start logging runs in MLFlow. Then set up another model and compare with LR baseline.


# 6/23
Had a persistent bug: A name collision between my Pipeline model and Linear Regression model, and was getting tangled up in why I was passing the incorrect arguments to my (ambiguously named) model. The correct approach is to include the lr model in the stages of the Pipeline.

I learned that a LogisticRegression object is a type of `Estimator` which may be added to a Pipeline. A pipeline can contain a model as a stage

# Monday 6/29
Encountered Query Plan Inflation, had to write training split to a table in the unity catalog.
Setting up MLFlow to track runs.
