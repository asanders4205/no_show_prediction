from pyspark.sql.types import IntegerType, BooleanType
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col,
    when,
    sum as spark_sum,
    monotonically_increasing_id,
)
import schemas
import yaml, os





def cast_numeric_columns(df):
    """Cast all integer and boolean columns to double for ML compatibility.

    Also explicitly casts the target column "Showed_up" to double.

    Args:
        df: Spark DataFrame with integer/boolean columns.

    Returns:
        Spark DataFrame with all numeric columns cast to double.
    """
    integer_cols = [
        c.name
        for c in df.schema.fields
        if isinstance(c.dataType, (IntegerType, BooleanType))
    ]

    for column in integer_cols:
        df = df.withColumn(column, col(column).cast("double"))

    df = df.withColumn("Showed_up", col("Showed_up").cast("double"))

    return df





def drop_null_columns(df):
    """Drop columns with more than 80% null values.

    Args:
        df: Spark DataFrame to audit.

    Returns:
        Spark DataFrame with high-null columns removed.
    """
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





def drop_null_records(df):
    """Drop rows with 3 or more null values.

    Args:
        df: Spark DataFrame to filter.

    Returns:
        Spark DataFrame with high-null rows removed.
    """
    NUM_COLS = len(df.columns)
    NULL_COLS_ALLOWED = 2
    THRESHOLD_VALUE = NUM_COLS - NULL_COLS_ALLOWED

    df = df.dropna(thresh=THRESHOLD_VALUE)

    return df




def add_unique_id(df):
    """Add a monotonically increasing unique ID column.

    Adds a "record_id" column starting at 1.  This can be configured as a
    primary key in Unity Catalog via a one-time DDL statement.

    Args:
        df: Spark DataFrame to augment.

    Returns:
        Spark DataFrame with a "record_id" column added.
    """
    df = df.withColumn("record_id", monotonically_increasing_id() + 1)

    return df





def validate_schema(df, expected_schema):
    """Validate that a DataFrame matches the expected schema.
    
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


# Entry point - only runs when executed directly, not when imported
if __name__ == "__main__":

    # Load from config file
    bronze_df = load_tbl_from_config("bronze_table_path")

    # Apply column renames to match expected schema
    column_renames = {
        "Neighbourhood": "Neighborhood",
        "Hipertension": "Hypertension",
        "Handcap": "Handicap",
        "Date.diff": "date_diff"
    }
    for old_name, new_name in column_renames.items():
        if old_name in bronze_df.columns:
            bronze_df = bronze_df.withColumnRenamed(old_name, new_name)
    
    # Cast PatientId to integer to match expected schema
    bronze_df = bronze_df.withColumn("PatientId", col("PatientId").try_cast("integer"))

    expected_patient_schema = schemas.PATIENT_SCHEMA

    # Validate schema (raises exception if invalid)
    validate_schema(bronze_df, expected_patient_schema)

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
    
    # Make ID Non-nullable, and set to primary key
    spark.sql("ALTER TABLE default.silver_no_show_features ALTER COLUMN record_id SET NOT NULL")
    spark.sql("ALTER TABLE default.silver_no_show_features ADD CONSTRAINT silver_pk PRIMARY KEY(record_id)")

