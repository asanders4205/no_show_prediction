''' Feature Engineering package. Modularization of feature_engineering Notebook'''



# Functions



# Imports
import math
from pyspark.sql.functions import col, sin, cos, month, dayofweek, dayofyear, lit


# Entry point
silver_df = spark.table("workspace.default.silver_no_show_features")










