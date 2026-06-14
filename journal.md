### Journal
#### Tuesday 6/9
  - Problem framing: Binary classification predicting Showed_up, no post-event leakage
  - Class imbalance: weightCol (~4× for no-shows) + AUC and Recall as primary metrics — motivated by the healthcare context where missing a no-show is costly
  - Feature engineering: Neighborhood encoded with TargetEncoder (not OHE — ~81 categories), date_diff added as a feature, column renames (Neighbourhood→Neighborhood,
  Hipertension→Hypertension) noted
  - Model progression: LogisticRegression (no tuning) → RandomForestClassifier (2×2 CV) → GBTClassifier (2×2 CV)
  - Tuning strategy: No tuning on LR baseline; CrossValidator 3-fold 2×2 grid on RF and GBT

  Still open: Gate 6 (evaluation metrics) — the recommendation was AUC + no-show Recall as headline numbers, logging all five metrics, but you hadn't confirmed that before we wrapped up.



