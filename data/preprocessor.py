# Modularization of the `preprocessing` Notebook

from pyspark.sql.types import IntegerType, BooleanType
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col,
    when,
    sum as spark_sum,
    monotonically_increasing_id,
) # End import
import schemas




"""cast_numeric_columns()
Purpose: Convert numeric (integer and numeric-typed boolean) columns to double
Output: TBD
"""
def cast_numeric_columns(df):
    # List integer and boolean columns - how to convert boolean and integer to double at once?
    integer_cols = [
        c.name
        for c in df.schema.fields
        if isinstance(c.dataType, (IntegerType, BooleanType))
    ]

    for column in integer_cols:
        df = df.withColumn(column, col(column).cast("double"))

    df = df.withColumn("Showed_up", col("Showed_up").cast("double"))

    return df



""" drop_null_columns()
    Purpose: Drop columns with > 80% (configurable) null values
    Return: Dataframe with null columns dropped
"""
def drop_null_columns(df):
    # Define threshold (e.g., 0.3 for 30%)
    threshold = 0.8
    total_rows = df.count()

    # Calculate null percentage for each column
    null_percentage = df.select(
        [
            (F.count(F.when(F.col(c).isNull(), c)) / total_rows).alias(c)
            for c in df.columns
        ]
    )

    # Identify columns exceeding the threshold
    cols_to_drop = [
        col
        for col in null_percentage.columns
        if null_percentage.first()[col] > threshold
    ]

    # Drop those columns
    df = df.drop(*cols_to_drop)

    return df



""" drop_null_records()
    Purpose: Remove rows with at least 3 null values
    Return: Dataframe with records dropped
"""
def drop_null_records(df):
    NUM_COLS = len(df.columns)
    NULL_COLS_ALLOWED = 3
    THRESHOLD_VALUE = NUM_COLS - NULL_COLS_ALLOWED

    df = df.dropna(thresh=THRESHOLD_VALUE)

    return df



""" add_unique_id()
    Purpose: Add monotonically increasing unique ID
    Return: Dataframe with ID column added
    
    Note: May need to configure use as primary key in the Unity Catalog. This is a one-time DDL statement
"""
def add_unique_id(df):
    # Add unique ID column
    df = df.withColumn("record_id", monotonically_increasing_id() + 1)

    return df



''' validate_schema()
    Purpose: Validate correctness of incoming data against established schema
    Return: Tuple (is_valid: bool, errors: list)
'''
def validate_schema(df, expected_schema):
    """
    Validates that the incoming DataFrame matches the expected schema.
    
    Args:
        df: Input PySpark DataFrame
        expected_schema: Expected StructType schema from schemas.py
    
    Returns:
        Tuple of (is_valid: bool, errors: list of error messages)
    """
    errors = []
    
    # Get actual schema
    actual_schema = df.schema
    actual_fields = {field.name: field for field in actual_schema.fields}
    expected_fields = {field.name: field for field in expected_schema.fields}
    
    # Check for missing columns
    missing_cols = set(expected_fields.keys()) - set(actual_fields.keys())
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")
    
    # Check for unexpected columns
    extra_cols = set(actual_fields.keys()) - set(expected_fields.keys())
    if extra_cols:
        errors.append(f"Unexpected columns found: {sorted(extra_cols)}")
    
    # Check data types for matching columns
    for col_name in expected_fields.keys():
        if col_name in actual_fields:
            expected_type = expected_fields[col_name].dataType
            actual_type = actual_fields[col_name].dataType
            
            if expected_type != actual_type:
                errors.append(
                    f"Column '{col_name}' type mismatch: "
                    f"expected {expected_type}, got {actual_type}"
                )
    
    # Check nullability for matching columns
    for col_name in expected_fields.keys():
        if col_name in actual_fields:
            expected_nullable = expected_fields[col_name].nullable
            actual_nullable = actual_fields[col_name].nullable
            
            # Only raise error if expected is non-nullable but actual is nullable
            if not expected_nullable and actual_nullable:
                errors.append(
                    f"Column '{col_name}' should be non-nullable but is nullable"
                )
    
    is_valid = len(errors) == 0
    
    # Print validation results
    if is_valid:
        print("✓ Schema validation passed successfully")
    else:
        print("✗ SCHEMA VALIDATION FAILED")
        print("=" * 60)
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
        print("=" * 60)
        # Raise exception to halt processing
        raise ValueError(
            f"Schema validation failed with {len(errors)} error(s). "
            "See above for details."
        )
    
    return is_valid, errors
    


# Declare bronze df
bronze_df = spark.table("workspace.default.bronze_features")

silver_df_validated = validate_schema(bronze_df,schemas.PATIENT_SCHEMA)

# Format bronze, convert to silver
silver_df_numerics_casted = cast_numeric_columns(bronze_df)

# Drop columns with many nulls
silver_df_nonulls = drop_null_columns(silver_df_numerics_casted)

# Add unique ID
silver_df = add_unique_id(silver_df_nonulls)

# Write to delta table
silver_df.write.mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("default.silver_no_show_features"
)