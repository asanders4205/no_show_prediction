''' Feature Engineering package. Modularization of feature_engineering Notebook'''


# Imports
import math
from pyspark.sql.functions import col, sin, cos, month, dayofweek, dayofyear, lit



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





# Entry point
silver_df = spark.table("workspace.default.silver_no_show_features")

train_df, test_df = train_test_split(silver_df)


# Base numeric features (cast to double, IDs and target excluded)
numerical_cols = ["Age", "Scholarship", "Hypertension", "Diabetes",
                  "Alcoholism", "Handicap", "SMS_received", "date_diff"]


numerical_cols, train_df, test_df = encode_cyclical_dates(numerical_cols, train_df, test_df)




























