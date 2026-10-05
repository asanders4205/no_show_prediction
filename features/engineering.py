import math
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, sin, cos, month, dayofyear, lit, create_map, when
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, MinMaxScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from pyspark.ml import Pipeline
from itertools import chain
import os, yaml
import matplotlib.pyplot as plt


def train_test_split(spark_df):
    """Split a Spark DataFrame into 80/20 train and test sets (seed=42).

    Args:
        spark_df: Spark DataFrame to split.

    Returns:
        train_df: Training DataFrame (~80%).
        test_df: Testing DataFrame (~20%).
    """
    train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train: {train_df.count():,}  Test: {test_df.count():,}")
    return train_df, test_df


def train_val_test_split(spark_df):
    """Split a Spark DataFrame into 60/20/20 train, validation, and test sets (seed=42).

    The validation set is used for threshold tuning; the test set is held out
    for final evaluation only.  All encoders and scalers are fit on the 60%
    training set to prevent leakage.

    Args:
        spark_df: Spark DataFrame to split.

    Returns:
        train_df: Training DataFrame (~60%).
        val_df: Validation DataFrame (~20%).
        test_df: Testing DataFrame (~20%).
    """
    train_df, val_df, test_df = spark_df.randomSplit([0.6, 0.2, 0.2], seed=42)
    print(f"Train: {train_df.count():,}  Val: {val_df.count():,}  Test: {test_df.count():,}")
    return train_df, val_df, test_df



def encode_cyclical_dates(numerical_cols, train_df, test_df, val_df=None):
    """Encode ScheduledDay and AppointmentDay as cyclical sin/cos features.

    Replaces the raw date columns with 8 derived features (month and
    day-of-year sin/cos pairs for each date).  Raw columns are dropped
    because VectorAssembler cannot handle date types.

    Args:
        numerical_cols: List of numeric feature column names to extend.
        train_df: Training Spark DataFrame with ScheduledDay and AppointmentDay.
        test_df: Testing Spark DataFrame with ScheduledDay and AppointmentDay.

    Returns:
        numerical_cols: Updated list with 8 cyclical features appended.
        train_df: Transformed training DataFrame.
        test_df: Transformed testing DataFrame.
    """

    dfs = [("train", train_df), ("test", test_df)]
    if val_df is not None:
        dfs.append(("val", val_df))

    transformed = {}
    for df_name, df in dfs:
        df = (df
            .withColumn("Sched_month_sin",     sin(lit(2 * math.pi) * month(col("ScheduledDay"))      / 12))
            .withColumn("Sched_month_cos",     cos(lit(2 * math.pi) * month(col("ScheduledDay"))      / 12))
            .withColumn("Sched_dayofyear_sin", sin(lit(2 * math.pi) * dayofyear(col("ScheduledDay"))  / 365))
            .withColumn("Sched_dayofyear_cos", cos(lit(2 * math.pi) * dayofyear(col("ScheduledDay"))  / 365))
            .withColumn("Appoi_month_sin",     sin(lit(2 * math.pi) * month(col("AppointmentDay"))    / 12))
            .withColumn("Appoi_month_cos",     cos(lit(2 * math.pi) * month(col("AppointmentDay"))    / 12))
            .withColumn("Appoi_dayofyear_sin", sin(lit(2 * math.pi) * dayofyear(col("AppointmentDay"))/ 365))
            .withColumn("Appoi_dayofyear_cos", cos(lit(2 * math.pi) * dayofyear(col("AppointmentDay"))/ 365))
            .drop("ScheduledDay", "AppointmentDay")   # raw date columns not usable by VectorAssembler
        )
        transformed[df_name] = df

    train_df = transformed["train"]
    test_df = transformed["test"]

    # Add the cyclical columns to the feature list defined in the cell above
    numerical_cols += [
        "Sched_month_sin", "Sched_month_cos",
        "Sched_dayofyear_sin", "Sched_dayofyear_cos",
        "Appoi_month_sin", "Appoi_month_cos",
        "Appoi_dayofyear_sin", "Appoi_dayofyear_cos",
    ]

    if val_df is not None:
        return numerical_cols, train_df, transformed["val"], test_df
    return numerical_cols, train_df, test_df


def neighborhood_encoder(numerical_cols, train_df, test_df, val_df=None):
    """Target-encode the Neighborhood column using sklearn TargetEncoder.

    Replaces each neighbourhood string with its mean no-show rate learned
    from the training set.  Fit on train only to prevent leakage.  The raw
    string column is dropped after encoding.

    Args:
        numerical_cols: List of numeric feature column names to extend.
        train_df: Training Spark DataFrame with a "Neighborhood" column.
        test_df: Testing Spark DataFrame with a "Neighborhood" column.

    Returns:
        numerical_cols: Updated list with "Neighborhood_te" appended.
        train_df: Transformed training DataFrame.
        test_df: Transformed testing DataFrame.
    """


    # 1. Collect training data to pandas — fit encoder on train only
    train_pd = train_df.select("Neighborhood", "Showed_up").toPandas()

    enc = TargetEncoder(target_type="binary", smooth="auto")
    enc.fit(train_pd[["Neighborhood"]], train_pd["Showed_up"])

    # 2. Build a lookup map: Neighborhood string - encoded float
    categories   = enc.categories_[0]                  # array of Neighborhood names
    encoded_vals = enc.transform(pd.DataFrame({"Neighborhood": categories}))

    mapping = dict(zip(categories, encoded_vals[:, 0].tolist()))

    # 3. Apply the map to both splits as a new Spark column
    map_expr = create_map([lit(x) for x in chain.from_iterable(mapping.items())])

    train_df = train_df.withColumn("Neighborhood_te", map_expr[col("Neighborhood")])
    test_df  = test_df.withColumn("Neighborhood_te", map_expr[col("Neighborhood")])

    # 4. Add to your numerical features and drop the raw string column
    numerical_cols += ["Neighborhood_te"]
    train_df = train_df.drop("Neighborhood")
    test_df  = test_df.drop("Neighborhood")

    if val_df is not None:
        val_df = val_df.withColumn("Neighborhood_te", map_expr[col("Neighborhood")])
        val_df = val_df.drop("Neighborhood")
        return numerical_cols, train_df, val_df, test_df
    return numerical_cols, train_df, test_df



def gender_encoder(numerical_cols):
    """Create StringIndexer and OneHotEncoder stages for the Gender column.

    Returns the stages unfitted so they can be fit on the training set later.

    Args:
        numerical_cols: List of numeric feature column names to extend.

    Returns:
        numerical_cols_with_gender: Updated list with "Gender_vec" appended.
        indexer: StringIndexer for "Gender" → "Gender_index".
        encoder: OneHotEncoder for "Gender_index" → "Gender_vec".
    """
    # Encode only the 'Gender' column
    indexer = StringIndexer(inputCol="Gender", outputCol="Gender_index", handleInvalid="skip")
    encoder = OneHotEncoder(inputCol="Gender_index", outputCol="Gender_vec")

    numerical_cols_with_gender = numerical_cols + ["Gender_vec"]

    return numerical_cols_with_gender, indexer, encoder



def fit_indexer_on_models(train_df, test_df, indexer, val_df=None):
    """Fit a StringIndexer on the training set and transform both splits.

    Args:
        train_df: Training Spark DataFrame.
        test_df: Testing Spark DataFrame.
        indexer: Unfitted StringIndexer instance.

    Returns:
        train_df_indexed: Transformed training DataFrame.
        test_df_indexed: Transformed testing DataFrame.
    """
    indexer_model = indexer.fit(train_df)
    train_df_indexed = indexer_model.transform(train_df)
    test_df_indexed = indexer_model.transform(test_df)

    if val_df is not None:
        val_df_indexed = indexer_model.transform(val_df)
        return train_df_indexed, val_df_indexed, test_df_indexed
    return train_df_indexed, test_df_indexed



def fit_encoder_on_models(train_df, test_df, encoder, val_df=None):
    """Fit an OneHotEncoder on the training set and transform both splits.

    Args:
        train_df: Training Spark DataFrame.
        test_df: Testing Spark DataFrame.
        encoder: Unfitted OneHotEncoder instance.

    Returns:
        train_df_encoded: Transformed training DataFrame.
        test_df_encoded: Transformed testing DataFrame.
    """
    # Fit encoder on train, transform all splits
    encoder_model = encoder.fit(train_df)
    train_df_encoded = encoder_model.transform(train_df)
    test_df_encoded = encoder_model.transform(test_df)

    if val_df is not None:
        val_df_encoded = encoder_model.transform(val_df)
        return train_df_encoded, val_df_encoded, test_df_encoded
    return train_df_encoded, test_df_encoded



def scale_datasets(train_df, test_df, val_df=None):
    """Apply MinMaxScaler to the "features" vector column of both splits.

    Scaler is fit on the training set only to prevent leakage.

    Args:
        train_df: Training Spark DataFrame with a "features" vector column.
        test_df: Testing Spark DataFrame with a "features" vector column.

    Returns:
        train_scaled: Scaled training DataFrame (adds "scaledFeatures").
        test_scaled: Scaled testing DataFrame (adds "scaledFeatures").
    """
    # Fit scaler on TRAIN only to prevent leakage
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    scalerModel = scaler.fit(train_df)


    # Transform all splits (scaler fit on train only)
    train_scaled = scalerModel.transform(train_df)
    test_scaled = scalerModel.transform(test_df)

    if val_df is not None:
        val_scaled = scalerModel.transform(val_df)
        return train_scaled, val_scaled, test_scaled
    return train_scaled, test_scaled



def apply_class_weights(train_scaled, test_scaled, val_df=None):
    """Add a weightCol to handle class imbalance in the training set.

    Computes show_count / no_show_count from the training data and assigns
    that ratio as the weight for no-shows (minority class).  Shows receive
    weight 1.0.

    Args:
        train_scaled: Training Spark DataFrame with a "Showed_up" column.
        test_scaled: Testing Spark DataFrame (returned unchanged).

    Returns:
        train_scaled_with_weight: Training DataFrame with "weightCol" added.
        test_scaled: Testing DataFrame (unchanged).
    """

    no_show_count = train_scaled.filter(col("Showed_up") == 0).count()
    show_count = train_scaled.filter(col("Showed_up") == 1).count()
    total_count = train_scaled.count()

    WEIGHT_SHOWED_UP = 1.0
    WEIGHT_NO_SHOW = show_count / no_show_count

    train_scaled_with_weight = train_scaled.withColumn(
        "weightCol",
        when(col("Showed_up") == 1, WEIGHT_SHOWED_UP).
        otherwise(WEIGHT_NO_SHOW)
    )

    if val_df is not None:
        # Val and test get weights too (needed for Spark ML evaluators that use weightCol)
        val_with_weight = val_df.withColumn(
            "weightCol",
            when(col("Showed_up") == 1, WEIGHT_SHOWED_UP).
            otherwise(WEIGHT_NO_SHOW)
        )
        test_with_weight = test_scaled.withColumn(
            "weightCol",
            when(col("Showed_up") == 1, WEIGHT_SHOWED_UP).
            otherwise(WEIGHT_NO_SHOW)
        )
        return train_scaled_with_weight, val_with_weight, test_with_weight
    return train_scaled_with_weight, test_scaled



def load_tbl_from_config(object_name):
    """Load a Spark DataFrame from a Unity Catalog table path in config.yaml.

    Looks for ../config.yaml first, falling back to config.example.yaml.

    Args:
        object_name: Key in the config dict (e.g. "bronze_table_path").

    Returns:
        Spark DataFrame loaded from the table at the configured path.
    """
    config_file = "../config.yaml" if os.path.exists("../config.yaml") else "config.example.yaml"

    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    table_path = cfg[object_name]

    return spark.table(table_path)





def determine_feature_importance(df):
    """Plot a correlation heatmap and per-feature |correlation with target|.

    Encodes categorical columns (Gender, Neighborhood, dates) into numeric
    representations so they can be included in the correlation matrix.

    Args:
        df: Spark or pandas DataFrame with features and "Showed_up" target.
    """
    pdf = df.toPandas() if hasattr(df, "toPandas") else df  # works for both Spark and pandas DataFrames

    # Encode Gender as numeric so it's included in the correlation matrix
    if "Gender" in pdf.columns and pdf["Gender"].dtype == object:
        pdf["Gender"] = pdf["Gender"].map({"F": 0, "M": 1})

    # Encode Neighborhood via target encoding so it's included in the correlation matrix
    if "Neighborhood" in pdf.columns and pdf["Neighborhood"].dtype == object:
        te = TargetEncoder(target_type="binary", smooth="auto")
        pdf["Neighborhood_te"] = te.fit_transform(pdf[["Neighborhood"]], pdf["Showed_up"])
        pdf = pdf.drop(columns=["Neighborhood"])

    # Encode date columns as cyclical sin/cos features so they're included in the correlation matrix
    for date_col, prefix in [("ScheduledDay", "Sched"), ("AppointmentDay", "Appoi")]:
        if date_col in pdf.columns:
            pdf[date_col] = pd.to_datetime(pdf[date_col])
            pdf[f"{prefix}_month_sin"] = np.sin(2 * math.pi * pdf[date_col].dt.month / 12)
            pdf[f"{prefix}_month_cos"] = np.cos(2 * math.pi * pdf[date_col].dt.month / 12)
            pdf[f"{prefix}_dayofyear_sin"] = np.sin(2 * math.pi * pdf[date_col].dt.dayofyear / 365)
            pdf[f"{prefix}_dayofyear_cos"] = np.cos(2 * math.pi * pdf[date_col].dt.dayofyear / 365)
            pdf = pdf.drop(columns=[date_col])

    # Correlation heatmap of all features + target
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = pdf.corr(numeric_only=True)
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix -1: Perfectly negative correlation, +1: Perfect positive correlation")
    plt.tight_layout()
    plt.show()

    # Bar chart of |correlation with target| for each feature
    target_corr = corr["Showed_up"].drop("Showed_up").abs().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    target_corr.plot.bar(ax=ax2, color="steelblue")
    ax2.set_title("|Correlation with Target| by Feature")
    ax2.set_ylabel("Absolute Pearson Correlation")
    ax2.axhline(y=0.05, color="red", linestyle="--", label="noise threshold (~0.05)")
    ax2.legend()
    plt.tight_layout()
    plt.show()







def recursive_feature_elimination(train_scaled, test_scaled, numerical_cols, n_features_to_select=5):
    """Use sklearn RFE to select the top N most important features.

    Collects scaled feature vectors and labels to pandas, expands the vector
    into individual columns, then runs RFE with a balanced-weights Logistic
    Regression as the base estimator.

    Args:
        train_scaled: Spark DataFrame with "scaledFeatures" vector and "Showed_up" label.
        test_scaled: Spark DataFrame with "scaledFeatures" vector and "Showed_up" label.
        numerical_cols: List of feature names corresponding to the vector elements.
        n_features_to_select: Number of features to keep (default 5).

    Returns:
        selected_features: List of selected feature names.
        X_train_rfe: pandas DataFrame with only the selected features (train).
        X_test_rfe: pandas DataFrame with only the selected features (test).
        y_train: pandas Series of training labels.
        y_test: pandas Series of test labels.
    """
    # Collect scaled feature vectors and labels to pandas
    train_pd = train_scaled.select("scaledFeatures", "Showed_up").toPandas()
    test_pd  = test_scaled.select("scaledFeatures", "Showed_up").toPandas()

    # Expand the DenseVector / SparseVector "scaledFeatures" into individual columns
    X_train = pd.DataFrame(train_pd["scaledFeatures"].tolist(), columns=numerical_cols)
    y_train = train_pd["Showed_up"]
    X_test  = pd.DataFrame(test_pd["scaledFeatures"].tolist(), columns=numerical_cols)
    y_test  = test_pd["Showed_up"]

    # Base estimator: sklearn LogisticRegression with class weights to handle imbalance
    estimator = SklearnLogisticRegression(max_iter=1000, class_weight="balanced")

    # Run Recursive Feature Elimination
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X_train, y_train)

    # Report results
    selected_features = [f for f, s in zip(numerical_cols, selector.support_) if s]
    print(f"\nRFE selected {len(selected_features)} of {len(numerical_cols)} features:\n")
    for i, (feat, rank) in enumerate(zip(numerical_cols, selector.ranking_)):
        marker = "  ✓ selected" if selector.support_[i] else f"  (rank {rank})"
        print(f"  {feat:30s}{marker}")

    # Transform both splits to selected features only
    X_train_rfe = pd.DataFrame(selector.transform(X_train), columns=selected_features)
    X_test_rfe  = pd.DataFrame(selector.transform(X_test),  columns=selected_features)

    return selected_features, X_train_rfe, X_test_rfe, y_train, y_test




# Entry point — only runs when executed directly, not when imported
if __name__ == "__main__":
    config_file = "../config.yaml" if os.path.exists("../config.yaml") else "config.example.yaml"
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    silver_df = load_tbl_from_config("silver_table_path")
    determine_feature_importance(silver_df)

    train_df, val_df, test_df = train_val_test_split(silver_df)

    # Base numeric features (cast to double, IDs and target excluded)
    numerical_cols = ["Age", "Scholarship", "Hypertension", "Diabetes",
                      "Alcoholism", "Handicap", "SMS_received", "date_diff"]


    # Encode cyclical dates (deterministic — no fitting, safe for all splits)
    numerical_cols, train_df, val_df, test_df = encode_cyclical_dates(numerical_cols, train_df, test_df, val_df)


    # Encode neighborhood (TargetEncoder fit on train only, applied to val+test)
    numerical_cols, train_df, val_df, test_df = neighborhood_encoder(numerical_cols, train_df, test_df, val_df)


    # Encode gender (StringIndexer/OneHotEncoder fit on train only, applied to val+test)
    numerical_cols, indexer, encoder = gender_encoder(numerical_cols)

    # Fit indexer
    train_df, val_df, test_df = fit_indexer_on_models(train_df, test_df, indexer, val_df)


    # Fit encoder
    train_df, val_df, test_df = fit_encoder_on_models(train_df, test_df, encoder, val_df)


    # Assemble features for all three splits
    vector_assembler = VectorAssembler(
        inputCols=numerical_cols,
        outputCol="features",
        handleInvalid="skip",
    )


    train_assembled = vector_assembler.transform(train_df)
    val_assembled = vector_assembler.transform(val_df)
    test_assembled = vector_assembler.transform(test_df)



    train_scaled, val_scaled, test_scaled = scale_datasets(train_assembled, test_assembled, val_df=val_assembled)

    # Handle class imbalance (weights computed from train only)
    train_scaled, val_scaled, test_scaled = apply_class_weights(train_scaled, test_scaled, val_df=val_scaled)

    # Write all three tables to Unity Catalog
    train_scaled.write.mode("overwrite").saveAsTable(cfg["train_scaled_dataset_path"])
    val_scaled.write.mode("overwrite").saveAsTable(cfg["val_scaled_dataset_path"])
    test_scaled.write.mode("overwrite").saveAsTable(cfg["test_scaled_dataset_path"])
    print(f"Tables written: train ({train_scaled.count():,}), val ({val_scaled.count():,}), test ({test_scaled.count():,})")
