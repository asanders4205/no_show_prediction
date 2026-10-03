import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, BooleanType
import yaml, os
import schemas


def load_csv_to_df(folder_path, dataset_name):
    """Load a CSV file into a Spark DataFrame using a predefined schema.

    Renames the 'Date.diff' column to 'date_diff' for downstream compatibility.

    Args:
        folder_path: Directory containing the CSV file.
        dataset_name: Name of the CSV file (e.g. "healthcare_noshows.csv").

    Returns:
        Spark DataFrame with the schema from schemas.PATIENT_SCHEMA applied.
    """
    dataset_path = f"{folder_path}{dataset_name}"
    pdf = pd.read_csv(dataset_path)

    loaded_spark_df = spark.read.format("csv") \
        .option("header", "true") \
        .schema(schemas.PATIENT_SCHEMA) \
        .load(dataset_path)

    processed_spark_df = loaded_spark_df.withColumnRenamed('Date.diff','date_diff')

    return processed_spark_df


def write_to_delta_table(spark_df, tbl_schema, tbl_name):
    """Write a Spark DataFrame to a Delta table in Unity Catalog.

    Overwrites the table and its schema if it already exists.

    Args:
        spark_df: Spark DataFrame to write.
        tbl_schema: Unity Catalog schema name (e.g. "default").
        tbl_name: Table name (e.g. "bronze_features").

    Returns:
        None
    """
    spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{tbl_schema}.{tbl_name}")
    return None





def get_dataset_path_yaml():
    """Return the raw dataset path from config.yaml.

    Looks for ../config.yaml first, falling back to config.example.yaml.

    Returns:
        The dataset_path string (a UC volume path to the raw CSV).
    """
    config_file = "../config.yaml" if os.path.exists("../config.yaml") else "config.example.yaml"

    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    dataset_path = cfg["dataset_path"]

    return dataset_path





# Entry point — only runs when executed directly, not when imported
if __name__ == "__main__":
    dataset_path = get_dataset_path_yaml()

    bronze_features_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(dataset_path)

    write_to_delta_table(bronze_features_df, "default", "bronze_features")





