# Project Handoff — No-Show Prediction

Last updated: Nov 12, 2026. Everything below reflects the live state as of this date (champions, tables, docs, and known issues).

**Start here for onboarding.** The other docs assume this context:

- `docs/README.md` — pipeline overview, metrics, configuration reference (updated Nov 12 for current champions)
- `docs/design.md` — business context, metric choice, cost asymmetry
- `docs/model_card_random_forest.md` / `docs/model_card_logistic_regression.md` — model cards (updated Nov 12, v22/v18)

---

## 1. What the project is

End-to-end ML pipeline on Databricks predicting patient appointment no-shows (Kaggle dataset, 110,527 raw records → 106,987 preprocessed). PySpark ML on serverless compute, MLflow tracking + Unity Catalog Model Registry, champion-alias promotion. Two models trained on identical splits and features: Logistic Regression (baseline) and Random Forest (champion). GBTClassifier is the planned next model.

## 2. Current state (Nov 12, 2026)

### Champions (both verified healthy: FINISHED runs, full params + metrics logged)

| | Random Forest **v22** (champion) | Logistic Regression **v18** (champion) |
|---|---|---|
| Registered model | `workspace.default.noshows_random_forest` | `workspace.default.noshows_logistic_regression` |
| Threshold (val-tuned) | 0.35 | 0.55 |
| PR-AUC | 0.9167 | 0.8879 |
| ROC-AUC | 0.7354 | 0.6839 |
| Accuracy | 0.7816 | 0.7504 |
| F1 (weighted) | 0.7561 | 0.7430 |
| No-show recall / precision / F1 | 0.2393 / 0.4315 / 0.3079 | 0.3133 / 0.3659 / 0.3376 |
| Test confusion (TP/FN/FP/TN) | 15,825 / 1,382 / 3,334 / 1,049 | 14,828 / 2,379 / 3,010 / 1,373 |
| Best params | numTrees 100, maxDepth 15, minInstancesPerNode 5, featureSubsetStrategy sqrt | elasticNetParam 0.0 (pure L2), regParam 0.001, maxIter 100 |
| Training time | ~13 min | ~5 min |

Note: LR v18 actually beats RF on no-show F1 (0.3376 vs 0.3079); RF leads on ranking quality (PR-AUC) and weighted F1.

### Data & splits (leak-free, regenerated Nov 11 2026)

- Tables: `workspace.default.no_show_{train,val,test}_scaled_tbl` — 64,176 / 21,221 / 21,590 rows, 30 columns
- `scaledFeatures`: 22 dims (Min-Max scaled). The prior 18 (demographics, conditions, `date_diff`, 8 cyclical date features, `Neighborhood_te`, `Gender_vec`) plus 4 point-in-time patient-history features: `prior_appointments`, `prior_no_shows`, `prior_no_show_ratio`, `has_history`
- History is computed on the **bronze** table via a window over `PatientId` ordered by `unix_date(AppointmentDay)` with `rangeBetween(unboundedPreceding, -1)` — strictly earlier appointment days only, so same-day and future outcomes never leak. ~40% of appointments have prior history. Validated brute-force against a self-join on all 106,987 rows: 0 mismatches
- Split is 60/20/20 (seed 42), performed **before** any encoder/scaler fitting (TargetEncoder, StringIndexer/OneHotEncoder, MinMaxScaler fit on train only). Thresholds are tuned on validation, never on test
- `weightCol` on all splits: no-show ~3.95 (class ratio), show 1.0. The LR notebook applies a milder square-root weighting (≈1.99)
- RFE: RF selects 12 of 22 (includes `prior_appointments`, `prior_no_shows`); LR selects 7 of 22 (a subset of RF's 12)

### Tests

49/49 passing (`tests/test_preprocessor.py` 15, `tests/test_engineering.py` 34 incl. 8 patient-history tests). Run procedure in §5.

## 3. What just happened (Nov 11–12, 2026)

1. **22-feature expansion**: added the 4 patient-history features to `engineering.py`, regenerated the three scaled tables, extended both notebooks' RFE feature lists, added 8 tests. RF marginal gains; LR huge gains (PR-AUC 0.7034 → 0.8879) — the earlier degenerate LR runs traced to training on a randomSplit-carved 48k subset, fixed by loading all splits from UC
2. **Three pipeline repairs** (job 68552777641482, two-task retrain job):
   - *SparseVector → pandas crash* in `recursive_feature_elimination` on both tasks: 22-dim vectors are stored sparse (~60% zeros from history features) and `toPandas()` treats SparseVector as an opaque scalar. Fixed by expanding via `v.toArray().tolist()`
   - *RF param logging*: `bestModel.getNumTrees` is an attribute on this runtime — calling it as a method raises TypeError
   - *LR orphan-champion bug*: `mlflow.spark.log_model` was called outside any active run, so registry versions pointed at empty permanently-RUNNING runs. Fixed by registering inside the explicit `mlflow.start_run` block
3. **Registry cleanup**: deleted LR versions v13–v17 and their five stuck source runs. v18 became champion (run 53efc45e, FINISHED, healthy)
4. **Docs**: README and both model cards fully updated for v22/v18, including confusion matrices, coefficients/importances extracted from the loaded champions, and corrected TargetEncoder semantics (see §4, first item)

## 4. Known issues & pending work

1. ~~`engineering.py` docstring bug~~ **FIXED Nov 12, 2026**: `neighborhood_encoder`'s docstring incorrectly said "mean no-show rate"; corrected to mean **show-up** rate (the TargetEncoder is fit on the `Showed_up` label) — README, model cards, and the module docstring now all agree
2. ~~Stale plot images~~ **FIXED Nov 12, 2026**: `docs/images/*.png` (ROC/PR curves, confusion matrices, feature importance, calibration) regenerated from the v22/v18 champions — rescored on the test table, all logged metrics reproduced exactly (confusion matrices, ROC-AUC, AP, accuracy, no-show precision/recall); README's stale-image note removed and its `date_diff` importance figure updated (0.594 → 0.342). Note: PR curves use showed-up (label 1) as the positive class, matching how the logged PR-AUC is calculated
3. **Git**: the leakage-fix branch was merged (PR #2, commit 717b211), but the Nov 11–12 changes (history feature, notebook updates, docs, tests) are **not yet committed**
4. ~~Log clutter~~ **FIXED Nov 12, 2026**: the two FAILED RandomForestModel MLflow runs from the numTrees crash (full run IDs `33a78f5d9339437fab03d667415f73c9`, `5d52ecdfea8644f8aed6b705c368bb5d`) were deleted; the experiment's active view now has 0 FAILED runs
5. **LR versions ≤12 reference runs in a deleted legacy experiment** — `get_run` fails for them; provenance is permanently broken (informational, they are not the champion)
6. ~~Stray empty files~~ **FIXED Nov 12, 2026**: the empty `CLAUDE.md` at the project root (created by mistake Nov 11) and the accidental empty copy at `/Users/asanders4205@gmail.com/docs/HANDOFF.md` (plus its stray parent `docs/` folder) were deleted; all removals verified gone via workspace status checks

## 5. Gotchas (all hit and solved — check here first)

- **Loading SparkML models on serverless**: `mlflow.spark.load_model` requires `os.environ["MLFLOW_DFS_TMP"] = <UC volume path>` (e.g. `cfg["volume_path"]`) on serverless/shared compute. Loaded champions come back as a `PipelineModel` wrapping a `CrossValidatorModel` — reach the classifier via `.stages[0].bestModel`
- **SparseVector → pandas**: see §3. Any code converting the 22-dim vectors with `toPandas()` must expand via `toArray()`
- **DATE → BIGINT casts are rejected on serverless**: use `unix_date()`. ANSI mode raises DIVIDE_BY_ZERO for division inside a filter `&` (use `try_divide`); division inside `when/otherwise` is safe
- **`np.trapz` is deprecated**: use `np.trapezoid` with argsorted recall for correct PR-AUC
- **ML session cache is 1GB**: call `setCollectSubModels(False)` on CrossValidator and clear model objects before fitting
- **Running pytest**: copy the project to /tmp first (pytest can't write `__pycache__` on the Workspace mount — OSError err95; the sh environment lacks SPARK_REMOTE); must include `data/` in the copy (tests import `preprocessor` from it); after re-staging, pop stale entries from `sys.path_importer_cache` before rerunning
- **`getNumTrees`** on RandomForestClassificationModel is an int attribute, not a method

## 6. Asset map

- **Project root**: `/Users/asanders4205@gmail.com/no_show_prediction/`
- **Config**: `config.yaml` (root, gitignored — each user creates their own) with keys: `dataset_path`, `volume_path`, `bronze_table_path`, `silver_table_path`, `train_scaled_dataset_path`, `val_scaled_dataset_path`, `test_scaled_dataset_path`, `lr_model_path`, `rf_model_path`, `features_path`, `mlflow.experiment_path`, `random_seed`. No workspace paths are hardcoded in committed code
- **Modules**: `features/engineering.py` (encoding, history, RFE, scaling), `data/loader.py`, `data/preprocessor.py`, `data/schemas.py`
- **Notebooks**: `models/random_forest` (notebook ID 1606283588053604), `models/logistic_regression` (notebook ID 1606283588053603) — both load all three splits from UC and register their champion inside the explicit MLflow run
- **Retrain job**: 68552777641482 (two notebook tasks). Nov 12 runs: 645808947315152 (RF, v22), 477604649397596 (LR, v18)
- **Registry**: `workspace.default.noshows_random_forest` (v22 @champion), `workspace.default.noshows_logistic_regression` (v18 @champion)
- **Experiment**: path defined in `mlflow.experiment_path` (config.yaml)

## 7. Verification playbook (how the Nov 12 numbers were validated)

The class-0 precision/recall and confusion matrices were derived by re-scoring the loaded champions on the test table and cross-checked against the logged MLflow metrics — weighted F1, PR-AUC, accuracy, and class-0 F1 all matched exactly (RF 0.7561/0.9167/0.7816/0.3079; LR 0.743/0.8879/0.7504/0.3376). To re-verify: load the champion, `VectorSlice` `scaledFeatures` to the model's selected feature indices (RF 12-of-22, LR 7-of-22) into `selectedFeatures`, transform the test table, apply the model's threshold (0.35 / 0.55) to `probability[1]`, and compare metrics with the run's logged values. Any drift means the tables or the champion changed.

---

*Prepared Nov 12, 2026. All metrics and facts above were verified directly against MLflow run data, the loaded champion models, and the UC tables on that date.*