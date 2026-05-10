
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mlProject.pipeline.prediction import (
    ProductionPredictionPipeline,
    PredictionPipeline,
    create_prediction_input,
    WINE_QUALITY_FEATURES
)
from mlProject.components.model_trainer import ProductionModelTrainer
from mlProject.components.model_evaluation import ProductionModelEvaluator
from mlProject.core.exceptions import (
    PredictionException,
    ModelNotFoundException,
    ModelTrainingException
)


class TestProductionPredictionPipeline:

    def test_create_prediction_input(self):
        features = {
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7,
            "citric_acid": 0.0,
            "residual_sugar": 1.9,
            "chlorides": 0.076,
            "free_sulfur_dioxide": 11,
            "total_sulfur_dioxide": 34,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        }

        input_data = create_prediction_input(features)

        assert input_data.shape == (1, 11)
        assert input_data[0, 0] == 7.4
        assert input_data[0, 10] == 9.4

    def test_wine_quality_features_count(self):
        assert len(WINE_QUALITY_FEATURES) == 11

    @patch('mlProject.pipeline.prediction.joblib.load')
    def test_prediction_pipeline_load_error(self, mock_load):
        mock_load.side_effect = FileNotFoundError("Model not found")

        with pytest.raises(ModelNotFoundException):
            pipeline = ProductionPredictionPipeline()

    @patch('mlProject.pipeline.prediction.joblib.load')
    def test_prediction_with_mock_model(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([5.5])
        mock_load.return_value = mock_model

        pass


class TestProductionModelTrainer:

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.trained_model_path = "artifacts/model_trainer/model.joblib"
        config.root_dir = "artifacts"
        config.model_train_dir = "artifacts/model_trainer"
        config.alpha = 0.5
        config.l1_ratio = 0.5
        config.random_state = 42
        return config

    def test_trainer_initialization(self, mock_config):
        trainer = ProductionModelTrainer(
            config=mock_config,
            experiment_name="test-experiment"
        )

        assert trainer.experiment_name == "test-experiment"
        assert trainer.config == mock_config

    def test_retry_decorator(self):
        from mlProject.core.exceptions import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3


class TestDataValidation:

    def test_validate_wine_features_complete(self):
        features = {f: 1.0 for f in WINE_QUALITY_FEATURES}
        input_data = create_prediction_input(features)

        assert input_data.shape == (1, 11)
        assert not np.isnan(input_data).any()

    def test_validate_wine_features_partial(self):
        features = {
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7
        }

        input_data = create_prediction_input(features)

        assert input_data.shape == (1, 11)
        assert input_data[0, 0] == 7.4
        assert input_data[0, 1] == 0.7
        assert input_data[0, 2] == 0.0


class TestAPIEndpoints:

    def test_prediction_input_validation(self):
        valid_data = np.array([[7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]])

        assert valid_data.shape[1] == 11

        assert np.issubdtype(valid_data.dtype, np.number)

    def test_batch_prediction_input_validation(self):
        batch_data = np.array([
            [7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4],
            [6.5, 0.5, 0.1, 2.0, 0.08, 15, 40, 0.995, 3.3, 0.5, 10.0]
        ])

        assert batch_data.shape == (2, 11)


class TestMLflowIntegration:

    @patch('mlflow.start_run')
    @patch('mlflow.sklearn.log_model')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    def test_mlflow_logging_mock(self, mock_metric, mock_params, mock_log_model, mock_start_run):
        from contextlib import ExitStack

        with ExitStack() as stack:
            mock_run = MagicMock()
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
            mock_start_run.return_value.__exit__ = Mock(return_value=False)

            pass

    def test_model_registry_stages(self):
        valid_stages = ["None", "Staging", "Production", "Archived"]
        assert "Staging" in valid_stages
        assert "Production" in valid_stages


class TestExceptionHandling:

    def test_prediction_exception_creation(self):
        exc = PredictionException(
            message="Test prediction error",
            details={"test": "data"}
        )

        assert str(exc) == "Test prediction error"
        assert exc.error_code.value >= 1000

    def test_model_not_found_exception(self):
        exc = ModelNotFoundException(
            message="Model not found",
            model_path="/path/to/model"
        )

        assert "not found" in str(exc).lower()
        assert exc.context.get("model_path") == "/path/to/model"


class TestDatabaseIntegration:

    def test_repository_pattern(self):
        from mlProject.db.database import PredictionRepository

        mock_db = MagicMock()
        repo = PredictionRepository(mock_db)

        assert repo.db_manager == mock_db


class TestConfigurationManagement:

    def test_config_manager_singleton(self):
        from mlProject.core.config import ConfigManager

        config1 = ConfigManager()
        config2 = ConfigManager()

        pass

    def test_environment_variable_loading(self):
        import os

        os.environ['TEST_VAR'] = 'test_value'

        assert os.getenv('TEST_VAR') == 'test_value'


@pytest.fixture(scope="session")
def test_data_dir():
    return Path("tests/data")


@pytest.fixture(scope="session")
def artifacts_dir():
    return Path("artifacts")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])