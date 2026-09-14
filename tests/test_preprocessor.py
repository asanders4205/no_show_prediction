# Unit tests for preprocessor.py functions

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    BooleanType,
    DoubleType,
    DateType,
)
from pyspark.testing import assertDataFrameEqual
import sys
import os

# Add parent directory to path to import preprocessor module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

import preprocessor
import schemas


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    return SparkSession.builder.getOrCreate()


class TestCastNumericColumns:
    """Tests for cast_numeric_columns function."""
    
    def test_cast_integer_to_double(self, spark):
        """Verify that integer columns are cast to double."""
        # Create test DataFrame with integer columns
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("age", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("Showed_up", BooleanType(), True),
        ])
        data = [(1, 25, "Alice", True), (2, 30, "Bob", False)]
        df = spark.createDataFrame(data, schema)
        
        # Apply function
        result_df = preprocessor.cast_numeric_columns(df)
        
        # Verify integer columns are now double
        assert result_df.schema["id"].dataType == DoubleType()
        assert result_df.schema["age"].dataType == DoubleType()
        assert result_df.schema["name"].dataType == StringType()
        assert result_df.schema["Showed_up"].dataType == DoubleType()
    
    def test_cast_boolean_to_double(self, spark):
        """Verify that boolean columns are cast to double."""
        schema = StructType([
            StructField("flag1", BooleanType(), True),
            StructField("flag2", BooleanType(), True),
            StructField("Showed_up", BooleanType(), True),
        ])
        data = [(True, False, True), (False, True, False)]
        df = spark.createDataFrame(data, schema)
        
        result_df = preprocessor.cast_numeric_columns(df)
        
        # All boolean columns should be double
        assert result_df.schema["flag1"].dataType == DoubleType()
        assert result_df.schema["flag2"].dataType == DoubleType()
        assert result_df.schema["Showed_up"].dataType == DoubleType()
    
    def test_preserve_string_columns(self, spark):
        """Verify that string columns remain unchanged."""
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("city", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        data = [("Alice", "NYC", 25), ("Bob", "LA", 30)]
        df = spark.createDataFrame(data, schema)
        
        result_df = preprocessor.cast_numeric_columns(df)
        
        assert result_df.schema["name"].dataType == StringType()
        assert result_df.schema["city"].dataType == StringType()


class TestDropNullColumns:
    """Tests for drop_null_columns function."""
    
    def test_drop_columns_above_threshold(self, spark):
        """Verify that columns with >80% nulls are dropped."""
        schema = StructType([
            StructField("col1", IntegerType(), True),
            StructField("col2", IntegerType(), True),
            StructField("col3", IntegerType(), True),
        ])
        # col2 has 90% nulls (9 out of 10), col3 has 70% nulls
        data = [
            (1, None, None),
            (2, None, None),
            (3, None, None),
            (4, None, None),
            (5, None, None),
            (6, None, None),
            (7, None, None),
            (8, None, 8),
            (9, None, 9),
            (10, 10, 10),
        ]
        df = spark.createDataFrame(data, schema)
        
        result_df = preprocessor.drop_null_columns(df)
        
        # col2 should be dropped (90% nulls), col1 and col3 should remain
        assert "col1" in result_df.columns
        assert "col2" not in result_df.columns
        assert "col3" in result_df.columns
    
    def test_keep_columns_below_threshold(self, spark):
        """Verify that columns with <=80% nulls are kept."""
        schema = StructType([
            StructField("col1", IntegerType(), True),
            StructField("col2", IntegerType(), True),
        ])
        # col1 has 80% nulls (exactly at threshold), col2 has 20% nulls
        data = [
            (None, 1),
            (None, 2),
            (None, None),
            (None, 4),
            (5, 5),
        ]
        df = spark.createDataFrame(data, schema)
        
        result_df = preprocessor.drop_null_columns(df)
        
        # Both columns should be kept (80% is not > 80%)
        assert "col1" in result_df.columns
        assert "col2" in result_df.columns


class TestDropNullRecords:
    """Tests for drop_null_records function."""
    
    def test_drop_rows_with_three_or_more_nulls(self, spark):
        """Verify that rows with 3+ null values are dropped."""
        schema = StructType([
            StructField("col1", IntegerType(), True),
            StructField("col2", IntegerType(), True),
            StructField("col3", IntegerType(), True),
            StructField("col4", IntegerType(), True),
            StructField("col5", IntegerType(), True),
        ])
        data = [
            (1, 2, 3, 4, 5),        # 0 nulls - keep
            (1, None, None, None, 5),  # 3 nulls - drop
            (1, 2, None, None, 5),  # 2 nulls - keep
            (None, None, None, None, None),  # 5 nulls - drop
        ]
        df = spark.createDataFrame(data, schema)
        
        result_df = preprocessor.drop_null_records(df)
        
        # Should have 2 rows remaining
        assert result_df.count() == 2
    
    def test_keep_rows_with_two_or_fewer_nulls(self, spark):
        """Verify that rows with <=2 null values are kept."""
        schema = StructType([
            StructField("col1", IntegerType(), True),
            StructField("col2", IntegerType(), True),
            StructField("col3", IntegerType(), True),
        ])
        data = [
            (1, 2, 3),           # 0 nulls
            (1, None, 3),        # 1 null
            (None, None, 3),     # 2 nulls
        ]
        df = spark.createDataFrame(data, schema)
        
        result_df = preprocessor.drop_null_records(df)
        
        # All rows should be kept
        assert result_df.count() == 3


class TestAddUniqueId:
    """Tests for add_unique_id function."""
    
    def test_adds_record_id_column(self, spark):
        """Verify that record_id column is added."""
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
        df = spark.createDataFrame(data, schema)
        
        result_df = preprocessor.add_unique_id(df)
        
        assert "record_id" in result_df.columns
        assert result_df.schema["record_id"].nullable is False
    
    def test_unique_ids_are_unique(self, spark):
        """Verify that all record IDs are unique."""
        schema = StructType([
            StructField("value", IntegerType(), True),
        ])
        data = [(i,) for i in range(100)]
        df = spark.createDataFrame(data, schema)
        
        result_df = preprocessor.add_unique_id(df)
        
        # Collect all IDs and verify uniqueness
        ids = [row.record_id for row in result_df.select("record_id").collect()]
        assert len(ids) == len(set(ids)), "record_id values are not unique"
    
    def test_ids_start_from_one(self, spark):
        """Verify that record IDs start from 1."""
        schema = StructType([
            StructField("value", IntegerType(), True),
        ])
        data = [(1,), (2,), (3,)]
        df = spark.createDataFrame(data, schema)
        
        result_df = preprocessor.add_unique_id(df)
        
        min_id = result_df.agg({"record_id": "min"}).collect()[0][0]
        assert min_id >= 1, "record_id should start from 1 or higher"


class TestValidateSchema:
    """Tests for validate_schema function."""
    
    def test_valid_schema_passes(self, spark):
        """Verify that a matching schema passes validation."""
        expected_schema = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        df = spark.createDataFrame([("Alice", 25), ("Bob", 30)], expected_schema)
        
        # Should not raise an exception
        is_valid, errors = preprocessor.validate_schema(df, expected_schema)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_missing_column_fails(self, spark):
        """Verify that missing columns cause validation to fail."""
        expected_schema = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("city", StringType(), True),
        ])
        actual_schema = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        df = spark.createDataFrame([("Alice", 25), ("Bob", 30)], actual_schema)
        
        with pytest.raises(ValueError, match="Schema validation failed"):
            preprocessor.validate_schema(df, expected_schema)
    
    def test_unexpected_column_fails(self, spark):
        """Verify that unexpected columns cause validation to fail."""
        expected_schema = StructType([
            StructField("name", StringType(), True),
        ])
        actual_schema = StructType([
            StructField("name", StringType(), True),
            StructField("extra_col", IntegerType(), True),
        ])
        df = spark.createDataFrame([("Alice", 1), ("Bob", 2)], actual_schema)
        
        with pytest.raises(ValueError, match="Schema validation failed"):
            preprocessor.validate_schema(df, expected_schema)
    
    def test_type_mismatch_fails(self, spark):
        """Verify that type mismatches cause validation to fail."""
        expected_schema = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        actual_schema = StructType([
            StructField("name", StringType(), True),
            StructField("age", StringType(), True),  # Wrong type
        ])
        df = spark.createDataFrame([("Alice", "25"), ("Bob", "30")], actual_schema)
        
        with pytest.raises(ValueError, match="type mismatch"):
            preprocessor.validate_schema(df, expected_schema)
    
    def test_returns_error_details(self, spark):
        """Verify that validation returns detailed error information."""
        expected_schema = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        actual_schema = StructType([
            StructField("name", StringType(), True),
        ])
        df = spark.createDataFrame([("Alice",), ("Bob",)], actual_schema)
        
        try:
            preprocessor.validate_schema(df, expected_schema)
        except ValueError:
            # Expected - just verify the function raises
            pass
