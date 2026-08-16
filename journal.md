### Journal

#### AI Disclaimer
I used Claude Code to write boilerplate code, provide direction, and help locate resources, up until the model logging portion of this project. Because this project is for deliberate practice and self-study, I used AI only in so far as I was comfortable with presenting this project as my own work.

In doing so I developed my skill of using AI as a learning resource instead of a crutch, while also accelerating development.

See 'Analysis of AI usage' for details



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


# 7/1
Working with the BinaryClassificationEvaluator() class.
Instantiate an object of this type, pass the Prediction columns, labels, and weights to it



Finished logging precision, recall, f1, accuracy, and auc with MLFlow. Getting more comfortable reading PySpark documentation.
Metrics logged in MLFlow for Logistic Regression and Random Forest. Super excited


# 7/6
There are residual errors. I started this project with Claude; using it as a learning aide, directing me where I was missing key pieces. I've removed all Claude-related files and am continuing by myself. I don't need claude to show me the way anymore. 

I'm logging my models in MLFlow. I have runs tracked, and LR is registered. Should I create feature tables? What about hyperparameter tuning? What else am I missing









### Analysis of AI usage:
Prompt: 'Summarize the purpose, methodology, and results of claude's use in this project'

  Searched for 6 patterns, read 3 files, ran 2 shell commands, recalled 1 memory

Claude's Use in This Project

Purpose: This is explicitly a learning project (per CLAUDE.md), not a ship-fast one. The goal is for you to learn to build an end-to-end PySpark ML pipeline on Databricks (patient no-show prediction) yourself, with Claude acting as a guide rather than an autonomous coder.

Methodology: Claude's role is constrained by two standing rules:
- CLAUDE.md requires pausing at key ML decision points (imbalance strategy, encoding choices, model selection, tuning strategy, evaluation metrics, interpretation) to explain concepts/tradeoffs before any code is written.
- A saved feedback memory reinforces this further: Claude should not silently write training/feature-engineering/evaluation code, and should favor Socratic questions over handing over snippets — explaining why, not what to type.

In practice, this shows up in journal.md as a series of dated sessions where you debugged issues yourself (e.g., a Pipeline vs. LogisticRegression naming collision, "Query Plan Inflation" requiring writes to Unity Catalog tables, correct use of BinaryClassificationEvaluator) with Claude apparently helping explain the underlying PySpark/MLflow concepts rather than fixing it directly.

Results so far (from git history, journal.md, goals.md):
- Problem framed as leakage-free binary classification on Showed_up.
- Feature engineering decided and implemented: TargetEncoder for high-cardinality Neighborhood, cyclical sin/cos date encoding, date_diff, column renames.
- Class imbalance addressed via a weightCol (~4× weight on no-shows).
- Logistic Regression baseline: built, evaluated, logged to MLflow (commit 6ed8ed9 LR logged).
- Random Forest: weight/ratio bugs fixed, trained with CV, evaluation metrics logged to MLflow (commit ffddb8d RF evaluation metrics logged) — most recent work.
- GBTClassifier is the planned next model per the architecture in CLAUDE.md/README.md, not yet started.
- Note: the actual AUC/F1/recall/precision numbers live in the MLflow experiment UI on Databricks — they aren't captured in local repo files, so I can't quote specific values from what's on disk. Also worth flagging: README.md/CLAUDE.md describe a production ml_pipeline_agent.py script, but the repo currently only contains the two notebooks — that script doesn't exist yet, so the "production pipeline" description is aspirational, not current state.


### 8/6
Acheived baseline accuracy of 66% and recall of 66% with Logistic Regression. Random Forest performs worse. I'm not tuning hyperparamters; I think that may be the next step. I have my runs registered in MLflow.

### 8/12
#### Baseline accuracy of 66% with Logistic Regression
Baseline accuracy was 66% for Logistic regression, I'm going to try to raise it by scaling the numeric features. I saw that Epic has a sandbox for accessing mock patient data, I may use it in a later phase to make the project more realistic and more applicable to actual clinical data.

#### Hyperparameter tuning
Scaling the numeric features didn't seem to improve accuracy or recall, I think I'll have to start tuning hyperparameters
* TBD which hyperparameters to tune
* Basic setup is complete




