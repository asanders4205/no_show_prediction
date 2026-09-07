
''' cast_numeric_columns()
    Purpose: Convert numeric (integer and numeric-typed boolean) columns to double
    Output: TBD
'''
def cast_numeric_columns(df):
    # List integer and boolean columns - how to convert boolean and integer to double at once?
    integer_cols = [
        c.name for c in df.schema.fields
        if isinstance(c.dataType, (IntegerType, BooleanType))
    ]

    for column in integer_cols:
        df = df.withColumn(column, col(column).cast("double"))

    df = df.withColumn("Showed_up", col("Showed_up").cast("double"))

    return df


''' drop_null_columns()
    Purpose: Drop columns with > 80% (configurable) null values
    Return: Dataframe with null columns dropped
'''
def drop_null_columns(df):
    # Define threshold (e.g., 0.3 for 30%)
    threshold = 0.8
    total_rows = df.count()

    # Calculate null percentage for each column
    null_percentage = df.select(
        [(F.count(F.when(F.col(c).isNull(), c)) / total_rows).alias(c) for c in df.columns]
    )

    # Identify columns exceeding the threshold
    cols_to_drop = [col for col in null_percentage.columns if null_percentage.first()[col] > threshold]

    # Drop those columns
    df = df.drop(*cols_to_drop)

    return df




''' drop_null_records()
    Purpose: Remove rows with at least 3 null values
    Return: Dataframe with records dropped
'''
def drop_null_records(df):
    NUM_COLS = len(df.columns)
    NULL_COLS_ALLOWED = 3
    THRESHOLD_VALUE = NUM_COLS - NULL_COLS_ALLOWED

    df = df.dropna(thresh=THRESHOLD_VALUE)

    return df


''' add_unique_id()
    Purpose: Add monotonically increasing unique ID
    Return: Dataframe with ID column added
    
    Note: May need to configure use as primary key in the Unity Catalog. This is a one-time DDL statement
'''
def add_unique_id(df):

    # Add unique ID column
    df = df.withColumn("record_id", monotonically_increasing_id() + 1)

    return df




from pyspark.sql.types import IntegerType, BooleanType
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, sum as spark_sum, monotonically_increasing_id


# Declare bronze df
bronze_df = spark.table("workspace.default.bronze_features")

# Format bronze, convert to silver
silver_df = cast_numeric_columns(bronze_df)

# Drop columns with many nulls
silver_df = drop_null_columns(silver_df)

# Add unique ID
silver_df = add_unique_id(silver_df)

# Write to delta table
silver_df.write.mode("overwrite").saveAsTable("default.silver_no_show_features")




