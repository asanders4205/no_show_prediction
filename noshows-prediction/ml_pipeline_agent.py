# Databricks notebook source
# An automated end-to-end ML pipeline: load → prep → feature engineering → train → MLflow log → register
# Run this notebook as a Databricks Job to retrain and promote a new model version automatically.

# COMMAND ----------
# MAGIC %md ## 0. Config

# COMMAND ----------

DATASET_PATH    = "/Workspace/Users/asanders4205@gmail.com/databricks_repo/noshows-prediction/input-datasets/healthcare_noshows.csv"
TARGET          = "Showed_up"
EXPERIMENT_NAME = "/Users/asanders4205@gmail.com/noshows-pipeline-agent"
MODEL_NAME      = "noshows_gbt"
RANDOM_SEED     = 42

# COMMAND ----------
# MAGIC %md ## 1. Load data with explicit schema

# COMMAND ----------

import math
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DateType, BooleanType
)
from pyspark.sql.functions import (
    col, when, sum as spark_sum, sin, cos, month, dayofweek, dayofyear, lit
)
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler
)
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

schema = StructType([
    StructField("PatientId",      IntegerType(), True),
    StructField("AppointmentID",  IntegerType(), True),
    StructField("Gender",         StringType(),  True),
    StructField("ScheduledDay",   DateType(),    True),
    StructField("AppointmentDay", DateType(),    True),
    StructField("Age",            IntegerType(), True),
    StructField("Neighbourhood",  StringType(),  True),
    StructField("Scholarship",    BooleanType(), True),
    StructField("Hipertension",   BooleanType(), True),
    StructField("Diabetes",       BooleanType(), True),
    StructField("Alcoholism",     BooleanType(), True),
    StructField("Handicap",       BooleanType(), True),
    StructField("SMS_received",   BooleanType(), True),
    StructField("Showed_up",      BooleanType(), True),
])

raw_df = (spark.read.format("csv")
          .option("header", "true")
          .option("nullValue", "null")
          .option("multiLine", "true")
          .schema(schema)
          .load(DATASET_PATH))

print(f"Loaded {raw_df.count():,} rows")

# COMMAND ----------
# MAGIC %md ## 2. Data preparation

# COMMAND ----------

from pyspark.sql.types import IntegerType, BooleanType

# Cast int/bool → double for Spark ML
numeric_bool_cols = [
    f.name for f in raw_df.schema.fields
    if isinstance(f.dataType, (IntegerType, BooleanType))
]
df = raw_df
for c in numeric_bool_cols:
    df = df.withColumn(c, col(c).cast("double"))

# Audit missing values — drop columns with > 60% nulls
n_rows = df.count()
missing = df.agg(*[
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
]).first().asDict()

high_null_cols = [c for c, v in missing.items() if v / n_rows > 0.6]
if high_null_cols:
    print(f"Dropping high-null columns: {high_null_cols}")
    df = df.drop(*high_null_cols)

# Drop ID columns before splitting
df = df.drop("PatientId", "AppointmentID")

train_df, test_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
n_train, n_test = train_df.count(), test_df.count()
print(f"Train: {n_train:,}  Test: {n_test:,}")

# COMMAND ----------
# MAGIC %md ## 3. Feature engineering

# COMMAND ----------

class CyclicalDateTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Adds sin/cos cyclical encodings for ScheduledDay and AppointmentDay, then drops
    the raw date columns. Included as a Pipeline stage so the logged model artifact
    handles raw date columns at inference time without manual preprocessing.
    """

    def _transform(self, df):
        for date_col in ["ScheduledDay", "AppointmentDay"]:
            if date_col not in df.columns:
                continue
            prefix = date_col[:5]
            m  = month(col(date_col))
            dy = dayofyear(col(date_col))
            dw = dayofweek(col(date_col))
            df = (df
                .withColumn(f"{prefix}_month_sin",     sin(lit(2 * math.pi) * m  / 12))
                .withColumn(f"{prefix}_month_cos",     cos(lit(2 * math.pi) * m  / 12))
                .withColumn(f"{prefix}_dayofyear_sin", sin(lit(2 * math.pi) * dy / 365))
                .withColumn(f"{prefix}_dayofyear_cos", cos(lit(2 * math.pi) * dy / 365))
                .withColumn(f"{prefix}_dayofweek_sin", sin(lit(2 * math.pi) * dw / 7))
                .withColumn(f"{prefix}_dayofweek_cos", cos(lit(2 * math.pi) * dw / 7))
                .drop(date_col))
        return df


# Explicit feature lists — avoids schema-introspection picking up the target or ID columns
categorical_cols = ["Gender", "Neighbourhood"]
numerical_cols = [
    # Original numeric columns (cast to double above)
    "Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handicap", "SMS_received",
    # Cyclical date features produced by CyclicalDateTransformer
    "Sched_month_sin",     "Sched_month_cos",
    "Sched_dayofyear_sin", "Sched_dayofyear_cos",
    "Sched_dayofweek_sin", "Sched_dayofweek_cos",
    "Appoi_month_sin",     "Appoi_month_cos",
    "Appoi_dayofyear_sin", "Appoi_dayofyear_cos",
    "Appoi_dayofweek_sin", "Appoi_dayofweek_cos",
]

indexers         = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="skip") for c in categorical_cols]
encoders         = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec", dropLast=True) for c in categorical_cols]
assembler_inputs = [f"{c}_vec" for c in categorical_cols] + numerical_cols
assembler        = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")

# COMMAND ----------
# MAGIC %md ## 4. Train with CrossValidator + MLflow

# COMMAND ----------

gbt = GBTClassifier(
    labelCol=TARGET,
    featuresCol="features",
    seed=RANDOM_SEED,
)

# CyclicalDateTransformer is the first stage so the serialised model handles raw
# ScheduledDay/AppointmentDay date columns at inference without external preprocessing.
full_pipeline = Pipeline(stages=[CyclicalDateTransformer()] + indexers + encoders + [assembler, gbt])

param_grid = (ParamGridBuilder()
    .addGrid(gbt.maxDepth, [3, 5])
    .addGrid(gbt.maxIter,  [20, 50])
    .build())

auc_evaluator = BinaryClassificationEvaluator(labelCol=TARGET, metricName="areaUnderROC")

cv = CrossValidator(
    estimator=full_pipeline,
    estimatorParamMaps=param_grid,
    evaluator=auc_evaluator,
    numFolds=3,
    seed=RANDOM_SEED,
)

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="gbt-cv-pipeline") as run:
    mlflow.log_params({
        "classifier":       "GBTClassifier",
        "maxDepth_grid":    "[3, 5]",
        "maxIter_grid":     "[20, 50]",
        "numFolds":         3,
        "train_rows":       n_train,      # reuse cached counts — no extra Spark scans
        "test_rows":        n_test,
        "categorical_cols": str(categorical_cols),
        "numerical_cols":   str(numerical_cols),
        "random_seed":      RANDOM_SEED,
    })

    cv_model    = cv.fit(train_df)
    best_model  = cv_model.bestModel
    predictions = best_model.transform(test_df)

    auc       = auc_evaluator.evaluate(predictions)
    acc       = MulticlassClassificationEvaluator(labelCol=TARGET, metricName="accuracy").evaluate(predictions)
    f1        = MulticlassClassificationEvaluator(labelCol=TARGET, metricName="f1").evaluate(predictions)
    precision = MulticlassClassificationEvaluator(labelCol=TARGET, metricName="weightedPrecision").evaluate(predictions)
    recall    = MulticlassClassificationEvaluator(labelCol=TARGET, metricName="weightedRecall").evaluate(predictions)

    mlflow.log_metrics({
        "test_auc":       auc,
        "test_accuracy":  acc,
        "test_f1":        f1,
        "test_precision": precision,
        "test_recall":    recall,
    })

    mlflow.log_dict(
        {"feature_columns": assembler_inputs, "target": TARGET},
        "feature_schema.json"
    )

    mlflow.spark.log_model(
        best_model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
    )

    run_id = run.info.run_id
    print(f"Run ID  : {run_id}")
    print(f"AUC     : {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1      : {f1:.4f}")

# COMMAND ----------
# MAGIC %md ## 5. Promote new version to champion alias

# COMMAND ----------

client = MlflowClient()

# Tie the version lookup to this run's ID to avoid a TOCTOU race with concurrent jobs
versions = client.search_model_versions(f"run_id='{run_id}'")
version  = versions[0].version

client.set_registered_model_alias(MODEL_NAME, "champion", version)
client.set_model_version_tag(MODEL_NAME, version, "run_id",       run_id)
client.set_model_version_tag(MODEL_NAME, version, "dataset",      "healthcare_noshows_v1")
client.set_model_version_tag(MODEL_NAME, version, "validated_by", "ml_pipeline_agent")

print(f"Model '{MODEL_NAME}' version {version} promoted to @champion")
print(f"Load with: mlflow.spark.load_model('models:/{MODEL_NAME}@champion')")
