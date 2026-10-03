"""Schema definitions and column-rename mappings for the no-show dataset."""

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, BooleanType, DoubleType

# Columns from the raw CSV that need renaming to match downstream expectations
RAW_COLUMN_RENAMES = {
    "Neighbourhood": "Neighborhood",
    "Hipertension": "Hypertension",
}

# Expected schema for the patient appointments dataset after column renames
PATIENT_SCHEMA = StructType([
    StructField("PatientId", IntegerType(), True),
    StructField("AppointmentID", IntegerType(), True),
    StructField("Gender", StringType(), True),
    StructField("ScheduledDay", DateType(), True),
    StructField("AppointmentDay", DateType(), True),
    StructField("Age", IntegerType(), True),
    StructField("Neighborhood", StringType(), True),
    StructField("Scholarship", BooleanType(), True),
    StructField("Hypertension", BooleanType(), True),
    StructField("Diabetes", BooleanType(), True),
    StructField("Alcoholism", BooleanType(), True),
    StructField("Handicap", BooleanType(), True),
    StructField("SMS_received", BooleanType(), True),
    StructField("Showed_up", BooleanType(), True),
    StructField("date_diff", IntegerType(), True),
])
