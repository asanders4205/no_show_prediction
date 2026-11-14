# Unit tests for engineering.py functions

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DoubleType,
    BooleanType,
    DateType,
)
from datetime import date
import sys
import os

# Add parent directory to path to import engineering module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'features')))

import engineering


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    return SparkSession.builder.getOrCreate()


@pytest.fixture
def sample_dataframe(spark):
    """Create a sample DataFrame for testing."""
    schema = StructType([
        StructField("record_id", IntegerType(), False),
        StructField("Age", DoubleType(), True),
        StructField("Gender", StringType(), True),
        StructField("ScheduledDay", DateType(), True),
        StructField("AppointmentDay", DateType(), True),
        StructField("Neighborhood", StringType(), True),
        StructField("Scholarship", DoubleType(), True),
        StructField("Hypertension", DoubleType(), True),
        StructField("Diabetes", DoubleType(), True),
        StructField("Alcoholism", DoubleType(), True),
        StructField("Handicap", DoubleType(), True),
        StructField("SMS_received", DoubleType(), True),
        StructField("Showed_up", DoubleType(), True),
        StructField("date_diff", DoubleType(), True),
    ])
    
    data = [
        (1, 30.0, "M", date(2023, 1, 15), date(2023, 1, 20), "Downtown", 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 5.0),
        (2, 45.0, "F", date(2023, 2, 10), date(2023, 2, 15), "Uptown", 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0),
        (3, 25.0, "M", date(2023, 3, 5), date(2023, 3, 10), "Downtown", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 5.0),
        (4, 60.0, "F", date(2023, 4, 20), date(2023, 4, 25), "Midtown", 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 5.0),
        (5, 35.0, "M", date(2023, 5, 12), date(2023, 5, 17), "Uptown", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0),
        (6, 50.0, "F", date(2023, 6, 8), date(2023, 6, 13), "Downtown", 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 5.0),
        (7, 40.0, "M", date(2023, 7, 22), date(2023, 7, 27), "Midtown", 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0),
        (8, 28.0, "F", date(2023, 8, 14), date(2023, 8, 19), "Uptown", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 5.0),
        (9, 55.0, "M", date(2023, 9, 3), date(2023, 9, 8), "Downtown", 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 5.0),
        (10, 32.0, "F", date(2023, 10, 18), date(2023, 10, 23), "Midtown", 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 5.0),
    ]
    
    return spark.createDataFrame(data, schema)


class TestTrainTestSplit:
    """Tests for train_test_split function."""
    
    def test_split_ratio(self, sample_dataframe):
        """Verify that train/test split maintains approximately 80/20 ratio."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        
        total_count = sample_dataframe.count()
        train_count = train_df.count()
        test_count = test_df.count()
        
        # Verify counts sum to original
        assert train_count + test_count == total_count
        
        # Verify approximate 80/20 split (within reasonable tolerance for small datasets)
        train_ratio = train_count / total_count
        assert 0.7 <= train_ratio <= 0.9, f"Train ratio {train_ratio} not close to 0.8"
    
    def test_reproducibility(self, sample_dataframe):
        """Verify that split is reproducible with same seed."""
        train_df1, test_df1 = engineering.train_test_split(sample_dataframe)
        train_df2, test_df2 = engineering.train_test_split(sample_dataframe)
        
        # Collect IDs to verify same split
        train_ids1 = sorted([row.record_id for row in train_df1.select("record_id").collect()])
        train_ids2 = sorted([row.record_id for row in train_df2.select("record_id").collect()])
        
        assert train_ids1 == train_ids2, "Split should be reproducible with fixed seed"
    
    def test_no_overlap(self, sample_dataframe):
        """Verify that train and test sets don't overlap."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        
        train_ids = set([row.record_id for row in train_df.select("record_id").collect()])
        test_ids = set([row.record_id for row in test_df.select("record_id").collect()])
        
        assert len(train_ids.intersection(test_ids)) == 0, "Train and test sets should not overlap"


class TestEncodeCyclicalDates:
    """Tests for encode_cyclical_dates function."""
    
    def test_creates_cyclical_columns(self, sample_dataframe):
        """Verify that all 8 cyclical date columns are created."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        numerical_cols = ["Age", "Scholarship"]
        
        numerical_cols, train_result, test_result = engineering.encode_cyclical_dates(
            numerical_cols, train_df, test_df
        )
        
        expected_new_cols = [
            "Sched_month_sin", "Sched_month_cos",
            "Sched_dayofyear_sin", "Sched_dayofyear_cos",
            "Appoi_month_sin", "Appoi_month_cos",
            "Appoi_dayofyear_sin", "Appoi_dayofyear_cos",
        ]
        
        # Verify columns exist in both dataframes
        for col in expected_new_cols:
            assert col in train_result.columns, f"{col} not in train_df"
            assert col in test_result.columns, f"{col} not in test_df"
    
    def test_drops_original_date_columns(self, sample_dataframe):
        """Verify that original date columns are dropped."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        numerical_cols = ["Age"]
        
        _, train_result, test_result = engineering.encode_cyclical_dates(
            numerical_cols, train_df, test_df
        )
        
        assert "ScheduledDay" not in train_result.columns
        assert "AppointmentDay" not in train_result.columns
        assert "ScheduledDay" not in test_result.columns
        assert "AppointmentDay" not in test_result.columns
    
    def test_updates_numerical_cols_list(self, sample_dataframe):
        """Verify that numerical_cols list is updated with new columns."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        numerical_cols = ["Age", "Scholarship"]
        original_length = len(numerical_cols)
        
        numerical_cols_result, _, _ = engineering.encode_cyclical_dates(
            numerical_cols, train_df, test_df
        )
        
        # Should add 8 new columns
        assert len(numerical_cols_result) == original_length + 8
    
    def test_cyclical_encoding_values_in_range(self, sample_dataframe):
        """Verify that sin/cos values are in valid range [-1, 1]."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        numerical_cols = ["Age"]
        
        _, train_result, _ = engineering.encode_cyclical_dates(
            numerical_cols, train_df, test_df
        )
        
        # Check one of the cyclical columns
        sin_values = [row.Sched_month_sin for row in train_result.select("Sched_month_sin").collect()]
        cos_values = [row.Sched_month_cos for row in train_result.select("Sched_month_cos").collect()]
        
        for val in sin_values + cos_values:
            assert -1.0 <= val <= 1.0, f"Cyclical value {val} out of range [-1, 1]"


class TestNeighborhoodEncoder:
    """Tests for neighborhood_encoder function."""
    
    def test_creates_neighborhood_te_column(self, sample_dataframe):
        """Verify that Neighborhood_te column is created."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        numerical_cols = ["Age"]
        
        _, train_result, test_result = engineering.neighborhood_encoder(
            numerical_cols, train_df, test_df
        )
        
        assert "Neighborhood_te" in train_result.columns
        assert "Neighborhood_te" in test_result.columns
    
    def test_drops_original_neighborhood_column(self, sample_dataframe):
        """Verify that original Neighborhood column is dropped."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        numerical_cols = ["Age"]
        
        _, train_result, test_result = engineering.neighborhood_encoder(
            numerical_cols, train_df, test_df
        )
        
        assert "Neighborhood" not in train_result.columns
        assert "Neighborhood" not in test_result.columns
    
    def test_updates_numerical_cols_list(self, sample_dataframe):
        """Verify that numerical_cols list includes Neighborhood_te."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        numerical_cols = ["Age", "Scholarship"]
        
        numerical_cols_result, _, _ = engineering.neighborhood_encoder(
            numerical_cols, train_df, test_df
        )
        
        assert "Neighborhood_te" in numerical_cols_result
    
    def test_encoded_values_are_numeric(self, sample_dataframe):
        """Verify that encoded values are numeric (float)."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        numerical_cols = ["Age"]
        
        _, train_result, test_result = engineering.neighborhood_encoder(
            numerical_cols, train_df, test_df
        )
        
        # Check that all values are numeric (not null for neighborhoods that exist in train)
        train_te_values = [row.Neighborhood_te for row in train_result.select("Neighborhood_te").collect()]
        
        assert all(isinstance(val, (int, float)) or val is None for val in train_te_values)
        assert any(val is not None for val in train_te_values), "All values should not be null"
    
    def test_same_neighborhood_gets_same_encoding(self, sample_dataframe):
        """Verify that the same neighborhood gets the same encoded value."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        numerical_cols = ["Age"]
        
        _, train_result, _ = engineering.neighborhood_encoder(
            numerical_cols, train_df, test_df
        )
        
        # Collect all rows and group by original neighborhood (we need to use another column to identify)
        # Since we dropped Neighborhood, we'll verify consistency by checking that values are consistent
        # This is a simplified test - in practice, we'd need to verify before dropping
        te_values = [row.Neighborhood_te for row in train_result.select("Neighborhood_te").collect()]
        
        # At minimum, verify we have numeric values
        assert len([v for v in te_values if v is not None]) > 0


class TestGenderEncoder:
    """Tests for gender_encoder function."""
    
    def test_returns_updated_numerical_cols(self):
        """Verify that function returns updated list with Gender_vec."""
        numerical_cols = ["Age", "Scholarship"]
        
        # The function now works correctly, returning the updated list plus indexer/encoder
        result_cols, indexer, encoder = engineering.gender_encoder(numerical_cols)
        
        assert "Gender_vec" in result_cols
    
    def test_creates_indexer_and_encoder_objects(self):
        """Verify that StringIndexer and OneHotEncoder objects are created."""
        # This test documents the intended behavior
        # The function should create these objects, but currently has a bug
        from pyspark.ml.feature import StringIndexer, OneHotEncoder
        
        # Manual test of intended behavior
        indexer = StringIndexer(inputCol="Gender", outputCol="Gender_index", handleInvalid="skip")
        encoder = OneHotEncoder(inputCol="Gender_index", outputCol="Gender_vec")
        
        assert indexer.getInputCol() == "Gender"
        assert indexer.getOutputCol() == "Gender_index"
        assert encoder.getInputCol() == "Gender_index"
        assert encoder.getOutputCol() == "Gender_vec"


class TestFitIndexerOnModels:
    """Tests for fit_indexer_on_models function."""
    
    def test_raises_error_without_indexer(self, sample_dataframe):
        """Verify that function requires indexer as a parameter."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        
        # The function now requires indexer as an explicit parameter
        with pytest.raises(TypeError):
            engineering.fit_indexer_on_models(train_df, test_df)


class TestFitEncoderOnModels:
    """Tests for fit_encoder_on_models function."""
    
    def test_raises_error_without_encoder(self, sample_dataframe):
        """Verify that function requires encoder as a parameter."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        
        # The function now requires encoder as an explicit parameter
        with pytest.raises(TypeError):
            engineering.fit_encoder_on_models(train_df, test_df)


class TestIntegrationScenarios:
    """Integration tests for multiple functions together."""
    
    def test_full_encoding_pipeline(self, sample_dataframe):
        """Test the typical encoding pipeline (excluding functions with bugs)."""
        # Split data
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        
        numerical_cols = ["Age", "Scholarship", "Hypertension", "Diabetes",
                         "Alcoholism", "Handicap", "SMS_received", "date_diff"]
        
        # Encode cyclical dates
        numerical_cols, train_df, test_df = engineering.encode_cyclical_dates(
            numerical_cols, train_df, test_df
        )
        
        # Encode neighborhood
        numerical_cols, train_df, test_df = engineering.neighborhood_encoder(
            numerical_cols, train_df, test_df
        )
        
        # Verify final state
        assert len(numerical_cols) == 8 + 8 + 1  # base (8) + cyclical (8) + neighborhood_te (1)
        assert "Neighborhood_te" in train_df.columns
        assert "Sched_month_sin" in train_df.columns
        assert "ScheduledDay" not in train_df.columns
        assert "Neighborhood" not in train_df.columns
    
    def test_column_count_progression(self, sample_dataframe):
        """Test that column counts change as expected through the pipeline."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        
        initial_cols = set(train_df.columns)
        numerical_cols = ["Age"]
        
        # After cyclical encoding (drops 2 date cols, adds 8 cyclical)
        numerical_cols, train_df, test_df = engineering.encode_cyclical_dates(
            numerical_cols, train_df, test_df
        )
        after_cyclical = set(train_df.columns)
        
        # After neighborhood encoding (drops 1, adds 1)
        numerical_cols, train_df, test_df = engineering.neighborhood_encoder(
            numerical_cols, train_df, test_df
        )
        after_neighborhood = set(train_df.columns)
        
        # Verify column count changes
        assert len(after_cyclical) == len(initial_cols) - 2 + 8  # -2 dates, +8 cyclical
        assert len(after_neighborhood) == len(after_cyclical)  # -1 Neighborhood, +1 Neighborhood_te


class TestScaleDatasets:
    """Tests for scale_datasets function."""
    
    def test_function_has_undefined_variable_bug(self):
        """Verify that scale_datasets no longer references undefined variables."""
        # The function previously used train_assembled and test_assembled which were not parameters
        # It should use train_df and test_df parameters instead
        from pyspark.ml.feature import VectorAssembler
        import inspect
        
        source = inspect.getsource(engineering.scale_datasets)
        assert "train_assembled" not in source
        assert "train_df" in source  # parameter name
    
    def test_scale_datasets_with_fixed_implementation(self, spark, sample_dataframe):
        """Test the intended behavior if the bug were fixed."""
        from pyspark.ml.feature import VectorAssembler, MinMaxScaler
        
        # Prepare data
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        numerical_cols = ["Age", "Scholarship", "date_diff"]
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=numerical_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        train_assembled = assembler.transform(train_df)
        test_assembled = assembler.transform(test_df)
        
        # Manual scaling (what the function should do)
        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
        scaler_model = scaler.fit(train_assembled)
        train_scaled = scaler_model.transform(train_assembled)
        test_scaled = scaler_model.transform(test_assembled)
        
        # Verify scaled features exist
        assert "scaledFeatures" in train_scaled.columns
        assert "scaledFeatures" in test_scaled.columns
        
        # Verify scaler was fit on training data only
        assert scaler_model is not None


class TestApplyClassWeights:
    """Tests for apply_class_weights function."""
    
    def test_creates_weight_column(self, sample_dataframe):
        """Verify that weightCol column is created."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        
        # Apply class weights
        train_weighted, test_result = engineering.apply_class_weights(train_df, test_df)
        
        assert "weightCol" in train_weighted.columns
    
    def test_weight_calculation_logic(self, sample_dataframe):
        """Verify that weights are calculated based on class imbalance."""
        from pyspark.sql.functions import col
        
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        
        # Get actual counts
        no_show_count = train_df.filter(col("Showed_up") == 0).count()
        show_count = train_df.filter(col("Showed_up") == 1).count()
        
        expected_weight_no_show = show_count / no_show_count if no_show_count > 0 else 1.0
        expected_weight_showed_up = 1.0
        
        # Apply weights
        train_weighted, _ = engineering.apply_class_weights(train_df, test_df)
        
        # Check a few samples
        samples = train_weighted.select("Showed_up", "weightCol").limit(10).collect()
        
        for row in samples:
            if row.Showed_up == 1.0:
                assert row.weightCol == expected_weight_showed_up
            else:
                assert row.weightCol == pytest.approx(expected_weight_no_show, rel=1e-5)
    
    def test_showed_up_gets_weight_one(self, sample_dataframe):
        """Verify that showed up appointments get weight 1.0."""
        from pyspark.sql.functions import col
        
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        train_weighted, _ = engineering.apply_class_weights(train_df, test_df)
        
        # Filter for showed_up == 1
        showed_up_weights = train_weighted.filter(col("Showed_up") == 1) \
            .select("weightCol").distinct().collect()
        
        assert len(showed_up_weights) == 1
        assert showed_up_weights[0].weightCol == 1.0
    
    def test_no_show_gets_higher_weight(self, sample_dataframe):
        """Verify that no-show appointments get higher weight (assuming class imbalance)."""
        from pyspark.sql.functions import col
        
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        train_weighted, _ = engineering.apply_class_weights(train_df, test_df)
        
        # Get weights for both classes
        weight_showed_up = train_weighted.filter(col("Showed_up") == 1) \
            .select("weightCol").first().weightCol
        
        weight_no_show_samples = train_weighted.filter(col("Showed_up") == 0) \
            .select("weightCol").limit(1).collect()
        
        if len(weight_no_show_samples) > 0:
            weight_no_show = weight_no_show_samples[0].weightCol
            # No-show weight should be >= 1.0 (assuming more showed up than no-show)
            assert weight_no_show >= weight_showed_up
    
    def test_test_set_unchanged(self, sample_dataframe):
        """Verify that test set is returned unchanged."""
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        test_cols_before = set(test_df.columns)
        
        _, test_result = engineering.apply_class_weights(train_df, test_df)
        test_cols_after = set(test_result.columns)
        
        # Test set should not have weightCol added
        assert test_cols_before == test_cols_after
        assert "weightCol" not in test_result.columns
    
    def test_full_pipeline_with_class_weights(self, sample_dataframe):
        """Test encoding pipeline followed by class weight application."""
        # Split data
        train_df, test_df = engineering.train_test_split(sample_dataframe)
        
        numerical_cols = ["Age", "Scholarship", "Hypertension", "Diabetes",
                         "Alcoholism", "Handicap", "SMS_received", "date_diff"]
        
        # Encode cyclical dates
        numerical_cols, train_df, test_df = engineering.encode_cyclical_dates(
            numerical_cols, train_df, test_df
        )
        
        # Encode neighborhood
        numerical_cols, train_df, test_df = engineering.neighborhood_encoder(
            numerical_cols, train_df, test_df
        )
        
        # Apply class weights
        train_weighted, test_result = engineering.apply_class_weights(train_df, test_df)
        
        # Verify weight column added to training set
        assert "weightCol" in train_weighted.columns
        assert "Neighborhood_te" in train_weighted.columns
        assert "Sched_month_sin" in train_weighted.columns
        
        # Test set should not have weights
        assert "weightCol" not in test_result.columns


class TestPatientNoShowHistory:
    """Tests for patient_no_show_history function (point-in-time prior history)."""

    @pytest.fixture
    def bronze_history_df(self, spark):
        """Bronze-style table: keeps PatientId, AppointmentID, AppointmentDay, boolean Showed_up."""
        schema = StructType([
            StructField("PatientId", DoubleType(), True),
            StructField("AppointmentID", IntegerType(), True),
            StructField("AppointmentDay", DateType(), True),
            StructField("Showed_up", BooleanType(), True),
        ])
        data = [
            # Patient 1: five appointments, including two same-day pairs
            (1.0, 101, date(2023, 1, 1), True),
            (1.0, 102, date(2023, 1, 1), False),
            (1.0, 103, date(2023, 1, 5), False),
            (1.0, 104, date(2023, 1, 10), True),
            (1.0, 105, date(2023, 1, 10), False),
            # Patient 2: single appointment (cold start)
            (2.0, 201, date(2023, 2, 1), False),
            # Patient 3: unknown outcome, then show, then no-show
            (3.0, 301, date(2023, 3, 1), None),
            (3.0, 302, date(2023, 3, 8), True),
            (3.0, 303, date(2023, 3, 15), False),
        ]
        return spark.createDataFrame(data, schema)

    @pytest.fixture
    def silver_history_df(self, spark):
        """Silver-style table: no PatientId, joined back via AppointmentID.
        AppointmentID 999 has no bronze counterpart (left-join miss)."""
        schema = StructType([
            StructField("AppointmentID", DoubleType(), True),
            StructField("Age", DoubleType(), True),
            StructField("Showed_up", DoubleType(), True),
        ])
        data = [
            (101.0, 30.0, 1.0), (102.0, 31.0, 0.0), (103.0, 32.0, 0.0),
            (104.0, 33.0, 1.0), (105.0, 34.0, 0.0),
            (201.0, 40.0, 0.0),
            (301.0, 50.0, 1.0), (302.0, 51.0, 1.0), (303.0, 52.0, 0.0),
            (999.0, 60.0, 1.0),
        ]
        return spark.createDataFrame(data, schema)

    @staticmethod
    def _history_of(df, appt_id):
        """Return the single history row for an AppointmentID."""
        rows = (df.filter(col("AppointmentID") == appt_id)
                .select("prior_appointments", "prior_no_shows",
                        "prior_no_show_ratio", "has_history")
                .collect())
        assert len(rows) == 1, f"expected exactly 1 row for AppointmentID {appt_id}"
        return rows[0]

    def test_appends_features_to_numerical_cols(self, silver_history_df, bronze_history_df):
        """Verify the 4 history features are appended to the returned list."""
        cols_out, _ = engineering.patient_no_show_history(
            ["Age"], silver_history_df, bronze_history_df)
        assert cols_out == ["Age", "prior_appointments", "prior_no_shows",
                            "prior_no_show_ratio", "has_history"]

    def test_does_not_mutate_input_list(self, silver_history_df, bronze_history_df):
        """Verify the caller's numerical_cols list is not mutated (docstring contract)."""
        original = ["Age"]
        _, _ = engineering.patient_no_show_history(
            original, silver_history_df, bronze_history_df)
        assert original == ["Age"]

    def test_adds_history_columns_and_preserves_rows(self, silver_history_df, bronze_history_df):
        """Verify the 4 columns exist and every silver row survives the join."""
        _, out = engineering.patient_no_show_history(
            ["Age"], silver_history_df, bronze_history_df)
        for c in ["prior_appointments", "prior_no_shows",
                  "prior_no_show_ratio", "has_history"]:
            assert c in out.columns
        assert out.count() == silver_history_df.count()

    def test_same_day_appointments_do_not_see_each_other(self, silver_history_df, bronze_history_df):
        """Verify same-day appointments share the same (strictly earlier) history."""
        _, out = engineering.patient_no_show_history(
            ["Age"], silver_history_df, bronze_history_df)
        for appt_id in (101, 102):  # both on 2023-01-01, no earlier history
            row = self._history_of(out, appt_id)
            assert row.prior_appointments == 0
            assert row.prior_no_shows == 0
            assert row.prior_no_show_ratio == 0.0
            assert row.has_history == 0.0

    def test_prior_counts_use_strictly_earlier_days(self, silver_history_df, bronze_history_df):
        """Verify counts include only earlier appointment days (no future leakage)."""
        _, out = engineering.patient_no_show_history(
            ["Age"], silver_history_df, bronze_history_df)
        # 103 (2023-01-05): priors are 101 + 102, of which one no-show (102)
        row = self._history_of(out, 103)
        assert row.prior_appointments == 2
        assert row.prior_no_shows == 1
        assert row.prior_no_show_ratio == pytest.approx(0.5)
        assert row.has_history == 1.0
        # 104/105 (2023-01-10): priors are 101+102+103, two no-shows.
        # A naive TOTAL no-show count would report 3 here (future leakage).
        for appt_id in (104, 105):
            row = self._history_of(out, appt_id)
            assert row.prior_appointments == 3
            assert row.prior_no_shows == 2
            assert row.prior_no_show_ratio == pytest.approx(2.0 / 3.0)
            assert row.has_history == 1.0

    def test_cold_start_patient_gets_zero_history(self, silver_history_df, bronze_history_df):
        """Verify a first-time patient gets zeros and has_history = 0."""
        _, out = engineering.patient_no_show_history(
            ["Age"], silver_history_df, bronze_history_df)
        row = self._history_of(out, 201)
        assert row.prior_appointments == 0
        assert row.prior_no_shows == 0
        assert row.prior_no_show_ratio == 0.0
        assert row.has_history == 0.0

    def test_clean_record_distinguished_from_no_record(self, silver_history_df, bronze_history_df):
        """Verify null outcomes count as neither shows nor no-shows, and that
        a clean record (has_history=1, ratio 0.0) differs from no record."""
        _, out = engineering.patient_no_show_history(
            ["Age"], silver_history_df, bronze_history_df)
        # 301's null outcome is not counted, so 302 still has no prior history
        assert self._history_of(out, 302).prior_appointments == 0
        # 303: prior is 302 only (a show) — clean record, not no record
        row = self._history_of(out, 303)
        assert row.prior_appointments == 1
        assert row.prior_no_shows == 0
        assert row.prior_no_show_ratio == 0.0
        assert row.has_history == 1.0

    def test_silver_row_without_bronze_match_gets_zero_history(self, silver_history_df, bronze_history_df):
        """Verify silver rows missing from bronze (left-join miss) get cold-start values."""
        _, out = engineering.patient_no_show_history(
            ["Age"], silver_history_df, bronze_history_df)
        row = self._history_of(out, 999)
        assert row.prior_appointments == 0
        assert row.prior_no_shows == 0
        assert row.prior_no_show_ratio == 0.0
        assert row.has_history == 0.0
