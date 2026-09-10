import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, BooleanType

import schemas



''' load_csv_to_df()
    Purpose: Reads in csv file path, returns Spark dataframe
    Return: Spark dataframe 
'''
def load_csv_to_df(folder_path, dataset_name):

    dataset_path = f"{folder_path}{dataset_name}"
    pdf = pd.read_csv(dataset_path)

    loaded_spark_df = spark.read.format("csv") \
        .option("header", "true") \
        .schema(schemas.PATIENT_SCHEMA) \
        .load(dataset_path)

    processed_spark_df = loaded_spark_df.withColumnRenamed('Date.diff','date_diff')

    return processed_spark_df


''' write_to_delta_table()
    Purpose:Write a spark df to a delta table
    Return: Void
'''
def write_to_delta_table(spark_df, tbl_schema, tbl_name):
    spark_df.write.mode("overwrite").saveAsTable(f"{tbl_schema}.{tbl_name}")


# Entry Point (only runs when executed directly, not when imported as a package)
if __name__ == "__main__":
    # dataset_folder = '/Workspace/Users/asanders4205@gmail.com/no_show_prediction/input-datasets/'
    dataset_folder = '/Workspace/Users/asanders4205@gmail.com/no_show_prediction/data/input-datasets/'
    appointment_noshow_dataset = "healthcare_noshows.csv"
    
    bronze_features_df = load_csv_to_df(dataset_folder, appointment_noshow_dataset)
    
    # Write to delta table
    write_to_delta_table(bronze_features_df, "default", "bronze_features")







