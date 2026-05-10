"""
Unit Tests for Production ML Pipeline
生产级ML管道单元测试
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import joblib
import json
import os
import tempfile

from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


class TestDataIngestion:
    """数据摄取测试"""

    def test_data_ingestion_config(self):
        """测试数据摄取配置"""
        from mlProject.core.config import DataIngestionConfig

        config = DataIngestionConfig(
            root_dir="test_artifacts/data_ingestion",
            source_url="https://example.com/data.zip",
            max_retries=5
        )

        assert config.root_dir == "test_artifacts/data_ingestion"
        assert config.max_retries == 5
        assert config.download_timeout == 300

    def test_data_ingestion_validation(self):
        """测试数据摄取验证"""
        from mlProject.components.data_ingestion import DataIngestion

        mock_config = Mock()
        mock_config.root_dir = tempfile.mkdtemp()
        mock_config.local_data_file = os.path.join(mock_config.root_dir, "test.zip")
        mock_config.unzip_dir = mock_config.root_dir

        os.makedirs(mock_config.root_dir, exist_ok=True)

        with open(mock_config.local_data_file, 'w') as f:
            f.write("test content")

        ingestion = DataIngestion(mock_config)

        assert os.path.exists(mock_config.root_dir)


class TestDataValidation:
    """数据验证测试"""

    def test_schema_validation(self):
        """测试模式验证"""
        from mlProject.components.data_validation import DataValidation

        temp_dir = tempfile.mkdtemp()
        test_csv = Path(temp_dir) / "test.csv"
        test_csv.write_text("col1,col2\n1,2\n3,4\n")

        mock_config = Mock()
        mock_config.root_dir = temp_dir
        mock_config.unzip_data_dir = str(test_csv)
        mock_config.status_file = os.path.join(temp_dir, "status.txt")

        validation = DataValidation(mock_config)
        result = validation.validate()

        assert result is not None


class TestDataTransformation:
    """数据转换测试"""

    def test_train_test_split(self):
        """测试训练测试集分割"""
        from sklearn.model_selection import train_test_split

        data = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'target': range(200, 300)
        })

        train, test = train_test_split(data, test_size=0.2, random_state=42)

        assert len(train) == 80
        assert len(test) == 20
        assert len(train) + len(test) == len(data)

    def test_feature_scaling(self):
        """测试特征缩放"""
        from sklearn.preprocessing import StandardScaler

        data = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        assert scaled.mean(axis=0).round().sum() == 0
        assert scaled.std(axis=0).round().sum() == 2


class TestModelTrainer:
    """模型训练测试"""

    def test_elastic_net_params(self):
        """测试ElasticNet参数"""
        from sklearn.linear_model import ElasticNet

        model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)

        assert model.alpha == 0.5
        assert model.l1_ratio == 0.5
        assert model.random_state == 42

    def test_model_training(self):
        """测试模型训练"""
        from sklearn.linear_model import ElasticNet
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        model = ElasticNet(alpha=0.1, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == 100
        assert not any(np.isnan(predictions))

    def test_model_persistence(self):
        """测试模型持久化"""
        from sklearn.linear_model import ElasticNet
        import tempfile
        import joblib

        model = ElasticNet(alpha=0.5, random_state=42)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        model.fit(X, y)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.joblib')
        joblib.dump(model, temp_file.name)

        loaded_model = joblib.load(temp_file.name)
        predictions = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(predictions, model.predict(X))

        os.unlink(temp_file.name)


class TestModelEvaluation:
    """模型评估测试"""

    def test_metrics_calculation(self):
        """测试指标计算"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        assert mse >= 0
        assert mae >= 0
        assert r2 <= 1
        assert r2 >= 0

    def test_rmse_calculation(self):
        """测试RMSE计算"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        assert rmse >= 0
        assert rmse < 1


class TestPredictionPipeline:
    """预测管道测试"""

    def test_single_prediction(self):
        """测试单个预测"""
        from sklearn.linear_model import ElasticNet

        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)

        model = ElasticNet(alpha=0.5, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.array([[1, 2, 3, 4, 5]])
        prediction = model.predict(X_test)

        assert isinstance(prediction[0], (float, np.floating))
        assert not np.isnan(prediction[0])

    def test_batch_prediction(self):
        """测试批量预测"""
        from sklearn.linear_model import ElasticNet

        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)

        model = ElasticNet(alpha=0.5, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(10, 5)
        predictions = model.predict(X_test)

        assert len(predictions) == 10
        assert not any(np.isnan(predictions))

    def test_prediction_shape(self):
        """测试预测形状"""
        from sklearn.linear_model import ElasticNet

        X_train = np.random.randn(100, 11)
        y_train = np.random.randn(100)

        model = ElasticNet(alpha=0.5, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(1, 11)
        prediction = model.predict(X_test)

        assert prediction.shape == (1,)


class TestExceptionHandling:
    """异常处理测试"""

    def test_base_exception_to_dict(self):
        """测试异常转字典"""
        from mlProject.core.exceptions import (
            BaseProductionException,
            ErrorCode
        )

        exc = BaseProductionException(
            message="Test error",
            error_code=ErrorCode.DATA_INGESTION_ERROR,
            details={"key": "value"}
        )

        error_dict = exc.to_dict()

        assert error_dict["error_code"] == 1001
        assert error_dict["message"] == "Test error"
        assert error_dict["details"]["key"] == "value"
        assert error_dict["retryable"] is True

    def test_model_not_found_exception(self):
        """测试模型未找到异常"""
        from mlProject.core.exceptions import ModelNotFoundException

        exc = ModelNotFoundException(
            message="Model not found",
            model_path="/path/to/model"
        )

        assert "model_path" in exc.details
        assert exc.details["model_path"] == "/path/to/model"

    def test_retry_decorator(self):
        """测试重试装饰器"""
        from mlProject.core.exceptions import retry_with_backoff

        counter = {"attempts": 0}

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky_function():
            counter["attempts"] += 1
            if counter["attempts"] < 3:
                raise ValueError("Temporary error")
            return "Success"

        result = flaky_function()

        assert result == "Success"
        assert counter["attempts"] == 3


class TestConfigManagement:
    """配置管理测试"""

    def test_config_dataclass(self):
        """测试配置数据类"""
        from mlProject.core.config import (
            DataIngestionConfig,
            ModelTrainerConfig,
            APIConfig
        )

        ingestion_config = DataIngestionConfig(
            root_dir="test",
            max_retries=5
        )

        trainer_config = ModelTrainerConfig(
            alpha=0.1,
            l1_ratio=0.5
        )

        api_config = APIConfig(
            host="localhost",
            port=8080
        )

        assert ingestion_config.max_retries == 5
        assert trainer_config.alpha == 0.1
        assert api_config.port == 8080

    def test_api_config_defaults(self):
        """测试API配置默认值"""
        from mlProject.core.config import APIConfig

        config = APIConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.debug is False
        assert config.workers == 4


class TestAPISchemas:
    """API模式测试"""

    def test_wine_quality_input_validation(self):
        """测试葡萄酒质量输入验证"""
        from mlProject.models.schemas import WineQualityInput

        valid_data = {
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

        input_data = WineQualityInput(**valid_data)

        assert input_data.fixed_acidity == 7.4
        assert input_data.alcohol == 9.4

    def test_prediction_response_schema(self):
        """测试预测响应模式"""
        from mlProject.models.schemas import PredictionResponse

        response = PredictionResponse(
            prediction=5.5,
            confidence=0.85,
            model_version="v1.0.0",
            model_type="elastic_net",
            inference_time_ms=12.5
        )

        assert response.prediction == 5.5
        assert response.model_version == "v1.0.0"

    def test_health_check_schema(self):
        """测试健康检查模式"""
        from mlProject.models.schemas import HealthCheckResponse

        response = HealthCheckResponse(
            status="healthy",
            version="2.0.0",
            uptime_seconds=3600.0,
            dependencies={"model": "healthy", "database": "healthy"}
        )

        assert response.status == "healthy"
        assert len(response.dependencies) == 2


class TestRateLimiter:
    """速率限制器测试"""

    def test_rate_limit_check(self):
        """测试速率限制检查"""
        from mlProject.middleware.auth import RateLimiter

        limiter = RateLimiter(requests_per_minute=10)

        for i in range(10):
            allowed, remaining = limiter.check_rate_limit("test_client")
            assert allowed is True

        allowed, remaining = limiter.check_rate_limit("test_client")
        assert allowed is False

    def test_rate_limit_refill(self):
        """测试速率限制补充"""
        from mlProject.middleware.auth import RateLimiter
        import time

        limiter = RateLimiter(requests_per_minute=60)

        limiter.check_rate_limit("test_client")
        allowed, remaining = limiter.check_rate_limit("test_client")

        assert remaining == 58


class TestTokenManager:
    """令牌管理器测试"""

    def test_create_access_token(self):
        """测试创建访问令牌"""
        from mlProject.middleware.auth import TokenManager

        manager = TokenManager()
        token = manager.create_access_token({"sub": "user123"})

        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_token(self):
        """测试验证令牌"""
        from mlProject.middleware.auth import TokenManager

        manager = TokenManager()
        token = manager.create_access_token({"sub": "user123"})

        payload = manager.verify_token(token)

        assert payload["sub"] == "user123"
        assert payload["type"] == "access"


class TestDatabaseModels:
    """数据库模型测试"""

    def test_prediction_model(self):
        """测试预测模型"""
        from mlProject.db.database import Prediction

        pred = Prediction(
            input_features="[1,2,3]",
            prediction=5.5,
            model_version="v1.0.0",
            inference_time_ms=10.5,
            status="success"
        )

        assert pred.prediction == 5.5
        assert pred.status == "success"

    def test_training_run_model(self):
        """测试训练运行模型"""
        from mlProject.db.database import TrainingRun

        start = datetime.utcnow()
        end = datetime.utcnow()

        run = TrainingRun(
            run_id="test_run_123",
            model_type="elastic_net",
            start_time=start,
            end_time=end,
            status="completed"
        )

        assert run.run_id == "test_run_123"
        assert run.status == "completed"


class TestLogging:
    """日志测试"""

    def test_structured_logger(self):
        """测试结构化日志"""
        from mlProject.middleware.logging import StructuredLogger
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = StructuredLogger(
                name="test_logger",
                log_dir=tmp_dir
            )

            logger.info("Test message")
            logger.error("Error message")

            log_file = Path(tmp_dir) / f"test_logger_{datetime.now().strftime('%Y%m%d')}.log"
            assert log_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
