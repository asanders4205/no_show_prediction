''' Feature Engineering package. Modularization of feature_engineering Notebook'''


# Imports
import math
from pyspark.sql.functions import col, sin, cos, month, dayofweek, dayofyear, lit
from sklearn.preprocessing import TargetEncoder
from pyspark.sql.functions import col, create_map, lit
from itertools import chain
import pandas as pd
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline

# Functions
''' train_test_split()
    Purpose: Run train test split, display counts
    Return:
        train_df - Training dataset
        test_df - Testing dataset
'''
def train_test_split(spark_df):
    train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train: {train_df.count():,}  Test: {test_df.count():,}")

    return train_df, test_df



''' encode_cyclical_dates()
    Purpose: Encode cyclical dates using cos(), sin()
    Parameters: train_df and test_df
    Return:
        - numerical_cols: Updated list of numerical columns (for scaling) 
        - train_df and test_df with date fields replaced with cyclical dates
'''
def encode_cyclical_dates(numerical_cols, train_df, test_df):

    for df_name, df in [("train", train_df), ("test", test_df)]:
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

        if df_name == "train":
            train_df = df
        else:
            test_df = df

    # Add the cyclical columns to the feature list defined in the cell above
    numerical_cols += [
        "Sched_month_sin", "Sched_month_cos",
        "Sched_dayofyear_sin", "Sched_dayofyear_cos",
        "Appoi_month_sin", "Appoi_month_cos",
        "Appoi_dayofyear_sin", "Appoi_dayofyear_cos",
    ]

    return numerical_cols, train_df, test_df


''' neighborhood_encoder()
    Purpose: Encode neighborhood in numerical vector
    Parameters:
        numerical_cols: List of numeric cols
        train_df
        test_df
    Return:
    # TODO fill
'''
def neighborhood_encoder(numerical_cols, train_df, test_df):


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

    return numerical_cols, train_df, test_df



''' gender_encoder() 
    Purpose: Encode gender to numeric vector
    Return: Numeric vector with values representing gender
'''
def gender_encoder(numerical_cols):
    # Encode only the 'Gender' column
    indexer = StringIndexer(inputCol="Gender", outputCol="Gender_index", handleInvalid="skip")
    encoder = OneHotEncoder(inputCol="Gender_index", outputCol="Gender_vec")

    numerical_cols_with_gender += ["Gender_vec"]

    return numerical_cols_with_gender



''' fit_indexer_on_models()
    Purpose: Fit indexer on train, transform both train and test
    Param:
    Return:
'''
def fit_indexer_on_models(train_df,test_df):
    indexer_model = indexer.fit(train_df)
    train_df_indexed = indexer_model.transform(train_df)
    test_df_indexed = indexer_model.transform(test_df)

    return train_df_indexed, test_df_indexed



''' fit_encoder_on_models()
    Purpose: Fit encoder on train, transform both train and test
    Param:
    Return:
'''
def fit_encoder_on_models(train_df,test_df):
    # Fit encoder on train, transform both train and test
    encoder_model = encoder.fit(train_df)
    train_df_encoded = encoder_model.transform(train_df)
    test_df_encoded = encoder_model.transform(test_df)

    return train_df_encoded, test_df_encoded





# Entry point
silver_df = spark.table("workspace.default.silver_no_show_features")

train_df, test_df = train_test_split(silver_df)

# TODO: These should not be hard coded
# Base numeric features (cast to double, IDs and target excluded)
numerical_cols = ["Age", "Scholarship", "Hypertension", "Diabetes",
                  "Alcoholism", "Handicap", "SMS_received", "date_diff"]


# Encode cyclical dates
numerical_cols, train_df, test_df = encode_cyclical_dates(numerical_cols, train_df, test_df)


# Encode neighborhood before scaling
numerical_cols, train_df, test_df = neighborhood_encoder(numerical_cols, train_df, test_df)


# Encode gender
numerical_cols = gender_encoder(numerical_cols)

# Fit indexer
train_df, test_df = fit_indexer_on_models(train_df,test_df)


# Fit encoder
train_df, test_df = fit_encoder_on_models(train_df,test_df)






















