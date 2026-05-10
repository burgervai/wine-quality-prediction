
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import json
from functools import lru_cache
from mlProject.core.exceptions import ConfigurationError


@dataclass
class DataIngestionConfig:
    root_dir: str = "artifacts/data_ingestion"
    source_url: str = "https://github.com/entbappy/Branching-tutorial/raw/master/winequality-data.zip"
    local_data_file: str = "artifacts/data_ingestion/data.zip"
    unzip_dir: str = "artifacts/data_ingestion"
    download_timeout: int = 300
    max_retries: int = 3


@dataclass
class DataValidationConfig:
    root_dir: str = "artifacts/data_validation"
    unzip_data_dir: str = "artifacts/data_ingestion/winequality-red.csv"
    status_file: str = "artifacts/data_validation/status.txt"
    schema_path: str = "schema.yaml"
    validation_batch_size: int = 1000


@dataclass
class DataTransformationConfig:
    root_dir: str = "artifacts/data_transformation"
    data_path: str = "artifacts/data_ingestion/winequality-red.csv"
    target_column: str = "quality"
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    imputation_strategy: str = "mean"
    scaling_method: str = "standard"
    train_test_split_ratio: float = 0.2


@dataclass
class ModelTrainerConfig:
    root_dir: str = "artifacts/model_trainer"
    train_data_path: str = "artifacts/data_transformation/train.csv"
    test_data_path: str = "artifacts/data_transformation/test.csv"
    model_name: str = "model.joblib"
    model_type: str = "elastic_net"
    alpha: float = 0.5
    l1_ratio: float = 0.5
    max_iter: int = 1000
    tol: float = 0.0001
    early_stopping: bool = True
    cv_folds: int = 5
    hyperparameter_tuning: bool = False


@dataclass
class ModelEvaluationConfig:

    root_dir: str = "artifacts/model_evaluation"
    test_data_path: str = "artifacts/data_transformation/test.csv"
    model_path: str = "artifacts/model_trainer/model.joblib"
    metric_file_name: str = "artifacts/model_evaluation/metrics.json"
    mlflow_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "wine-quality-prediction"
    threshold_rmse: float = 0.5
    threshold_r2: float = 0.6
    target_column: str = "quality"
    all_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIConfig:
 
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    workers: int = 4
    reload: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit: str = "100/minute"
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    timeout: int = 30


@dataclass
class LoggingConfig:

    level: str = "INFO"
    format: str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
    log_dir: str = "logs"
    log_file: str = "running_logs.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    log_to_console: bool = True
    log_to_file: bool = True


@dataclass
class DatabaseConfig:
  
    host: str = "localhost"
    port: int = 5432
    database: str = "ml_pipeline"
    user: str = "postgres"
    password: str = ""
    connection_pool_size: int = 10
    connection_timeout: int = 30
    use_sqlite: bool = True
    sqlite_path: str = "data/ml_pipeline.db"


@dataclass
class MonitoringConfig:
   
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_path: str = "/health"
    metrics_path: str = "/metrics"
    enable_tracing: bool = False
    trace_sample_rate: float = 0.1


class ConfigManager:
  
    _instance: Optional['ConfigManager'] = None
    _config_cache: Dict[str, Any] = {}

    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config_dir = Path("config")
        self._config_file = self._config_dir / "config.yaml"
        self._env_file = self._config_dir / ".env"
        self._load_env_variables()

    def _load_env_variables(self) -> None:
        
        if self._env_file.exists():
            with open(self._env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
     
        if config_path:
            path = Path(config_path)
        else:
            path = self._config_file

        if not path.exists():
            raise ConfigurationError(f"配置文件不存在: {path}")

        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)

        self._config_cache = config_data
        return config_data

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        
        config = self._config_cache.get('data_ingestion', {})
        return DataIngestionConfig(
            root_dir=config.get('root_dir', 'artifacts/data_ingestion'),
            source_url=config.get('source_URL', ''),
            local_data_file=config.get('local_data_file', ''),
            unzip_dir=config.get('unzip_dir', ''),
            download_timeout=config.get('download_timeout', 300),
            max_retries=config.get('max_retries', 3)
        )

    def get_data_validation_config(self) -> DataValidationConfig:
      
        config = self._config_cache.get('data_validation', {})
        return DataValidationConfig(
            root_dir=config.get('root_dir', 'artifacts/data_validation'),
            unzip_data_dir=config.get('unzip_data_dir', ''),
            status_file=config.get('STATUS_FILE', ''),
            schema_path=config.get('schema_path', 'schema.yaml'),
            validation_batch_size=config.get('validation_batch_size', 1000)
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
       
        config = self._config_cache.get('data_transformation', {})
        return DataTransformationConfig(
            root_dir=config.get('root_dir', 'artifacts/data_transformation'),
            data_path=config.get('data_path', ''),
            target_column=config.get('target_column', 'quality'),
            categorical_columns=config.get('categorical_columns', []),
            numerical_columns=config.get('numerical_columns', []),
            imputation_strategy=config.get('imputation_strategy', 'mean'),
            scaling_method=config.get('scaling_method', 'standard'),
            train_test_split_ratio=config.get('train_test_split_ratio', 0.2)
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        
        config = self._config_cache.get('model_trainer', {})
        return ModelTrainerConfig(
            root_dir=config.get('root_dir', 'artifacts/model_trainer'),
            train_data_path=config.get('train_data_path', ''),
            test_data_path=config.get('test_data_path', ''),
            model_name=config.get('model_name', 'model.joblib'),
            model_type=config.get('model_type', 'elastic_net'),
            alpha=config.get('alpha', 0.5),
            l1_ratio=config.get('l1_ratio', 0.5),
            max_iter=config.get('max_iter', 1000),
            tol=config.get('tol', 0.0001),
            early_stopping=config.get('early_stopping', True),
            cv_folds=config.get('cv_folds', 5),
            hyperparameter_tuning=config.get('hyperparameter_tuning', False)
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        
        config = self._config_cache.get('model_evaluation', {})
        return ModelEvaluationConfig(
            root_dir=config.get('root_dir', 'artifacts/model_evaluation'),
            test_data_path=config.get('test_data_path', ''),
            model_path=config.get('model_path', ''),
            metric_file_name=config.get('metric_file_name', ''),
            mlflow_uri=os.environ.get('MLFLOW_TRACKING_URI', config.get('mlflow_uri', 'http://localhost:5000')),
            mlflow_experiment_name=config.get('mlflow_experiment_name', 'wine-quality-prediction'),
            threshold_rmse=config.get('threshold_rmse', 0.5),
            threshold_r2=config.get('threshold_r2', 0.6),
            target_column=config.get('target_column', 'quality'),
            all_params=config.get('all_params', {})
        )

    def get_api_config(self) -> APIConfig:
       
        return APIConfig(
            host=os.environ.get('API_HOST', '0.0.0.0'),
            port=int(os.environ.get('API_PORT', '8080')),
            debug=os.environ.get('DEBUG', 'false').lower() == 'true',
            workers=int(os.environ.get('WORKERS', '4')),
            reload=os.environ.get('RELOAD', 'false').lower() == 'true',
            cors_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
            rate_limit=os.environ.get('RATE_LIMIT', '100/minute'),
            max_request_size=int(os.environ.get('MAX_REQUEST_SIZE', str(16 * 1024 * 1024))),
            timeout=int(os.environ.get('TIMEOUT', '30'))
        )

    def get_logging_config(self) -> LoggingConfig:
      
        config = self._config_cache.get('logging', {})
        return LoggingConfig(
            level=config.get('level', 'INFO'),
            format=config.get('format', '[%(asctime)s: %(levelname)s: %(module)s: %(message)s]'),
            log_dir=config.get('log_dir', 'logs'),
            log_file=config.get('log_file', 'running_logs.log'),
            max_bytes=config.get('max_bytes', 10 * 1024 * 1024),
            backup_count=config.get('backup_count', 5),
            log_to_console=config.get('log_to_console', True),
            log_to_file=config.get('log_to_file', True)
        )

    def get_database_config(self) -> DatabaseConfig:
       
        return DatabaseConfig(
            host=os.environ.get('DB_HOST', 'localhost'),
            port=int(os.environ.get('DB_PORT', '5432')),
            database=os.environ.get('DB_NAME', 'ml_pipeline'),
            user=os.environ.get('DB_USER', 'postgres'),
            password=os.environ.get('DB_PASSWORD', ''),
            connection_pool_size=int(os.environ.get('DB_POOL_SIZE', '10')),
            connection_timeout=int(os.environ.get('DB_TIMEOUT', '30')),
            use_sqlite=os.environ.get('USE_SQLITE', 'true').lower() == 'true',
            sqlite_path=os.environ.get('SQLITE_PATH', 'data/ml_pipeline.db')
        )

    def get_monitoring_config(self) -> MonitoringConfig:
        
        config = self._config_cache.get('monitoring', {})
        return MonitoringConfig(
            enable_metrics=config.get('enable_metrics', True),
            metrics_port=config.get('metrics_port', 9090),
            health_check_path=config.get('health_check_path', '/health'),
            metrics_path=config.get('metrics_path', '/metrics'),
            enable_tracing=config.get('enable_tracing', False),
            trace_sample_rate=config.get('trace_sample_rate', 0.1)
        )


@lru_cache()

    manager = ConfigManager()
    manager.load_config()
    return manager
