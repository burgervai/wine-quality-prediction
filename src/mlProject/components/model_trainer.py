"""
Production Model Trainer with MLflow Integration

"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import warnings

from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.core.exceptions import ModelTrainingException
from mlProject.core.exceptions import retry_with_backoff


class ProductionModelTrainer:
    """生产级模型训练器 - 集成MLflow实验跟踪"""

    def __init__(
        self,
        config: ModelTrainerConfig,
        experiment_name: str = "wine-quality-prediction",
        tracking_uri: Optional[str] = None
    ):
        
        self.config = config
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000"
        )
        self.run_id = None
        self.model = None
        self.metrics = {}

    def _setup_mlflow(self) -> None:
       
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow已配置: {self.tracking_uri}")

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def train(
        self,
        hyperparameters: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        enable_logging: bool = True
    ) -> Dict[str, Any]:
      
        try:
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            train_x = train_data.drop([self.config.target_column], axis=1)
            test_x = test_data.drop([self.config.target_column], axis=1)
            train_y = train_data[[self.config.target_column]]
            test_y = test_data[[self.config.target_column]]

            params = self._get_hyperparameters(hyperparameters)

            if enable_logging:
                return self._train_with_mlflow(
                    train_x, train_y, test_x, test_y, params, run_name
                )
            else:
                return self._train_without_mlflow(train_x, train_y, params)

        except Exception as e:
            logger.error(f": {str(e)}")
            raise ModelTrainingException(
                message=f": {str(e)}",
                cause=e
            )

    def _get_hyperparameters(
        self,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
       
        params = {
            "alpha": self.config.alpha,
            "l1_ratio": self.config.l1_ratio,
            "max_iter": getattr(self.config, 'max_iter', 1000),
            "tol": getattr(self.config, 'tol', 0.0001),
            "random_state": 42
        }

        if overrides:
            params.update(overrides)

        return params

    def _train_with_mlflow(
        self,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        test_x: pd.DataFrame,
        test_y: pd.DataFrame,
        params: Dict[str, Any],
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """使用MLflow进行训练和日志记录"""
        self._setup_mlflow()

        with mlflow.start_run(run_name=run_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            self.run_id = run.info.run_id
            logger.info(f": {self.run_id}")

            mlflow.log_params(params)

            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True
            )

            self.model = self._create_model(params)

            logger.info("")
            start_time = datetime.now()

            self.model.fit(train_x, train_y)

            training_time = (datetime.now() - start_time).total_seconds()
            mlflow.log_param("training_time_seconds", training_time)

            predictions = self.model.predict(test_x)

            signature = infer_signature(test_x, predictions)
            mlflow.sklearn.log_model(
                self.model,
                "model",
                signature=signature,
                registered_model_name=self.config.model_name.replace(".joblib", "")
            )

            self.metrics = self._calculate_metrics(test_y, predictions)
            for metric_name, metric_value in self.metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            self._save_model()

            result = {
                "run_id": self.run_id,
                "status": "success",
                "metrics": self.metrics,
                "training_time_seconds": training_time,
                "model_path": os.path.join(self.config.root_dir, self.config.model_name),
                "artifact_uri": run.info.artifact_uri
            }

            logger.info(f"训练完成 - 指标: {self.metrics}")
            return result

    def _train_without_mlflow(
        self,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
      
        self.model = self._create_model(params)
        self.model.fit(train_x, train_y)
        self._save_model()

        return {
            "status": "success",
            "model_path": os.path.join(self.config.root_dir, self.config.model_name)
        }

    def _create_model(self, params: Dict[str, Any]):
       
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**params)

    def _calculate_metrics(
        self,
        y_true: pd.DataFrame,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
      
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score
        )

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    def _save_model(self) -> str:
     
        os.makedirs(self.config.root_dir, exist_ok=True)
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(self.model, model_path)
        logger.info(f"模型已保存: {model_path}")
        return model_path

    def load_model(self) -> Any:
      
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.model = joblib.load(model_path)
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
      
        if self.model is None:
            self.load_model()
        return self.model.predict(X)

    def register_model(
        self,
        model_name: str,
        stage: str = "Staging",
        description: str = ""
    ) -> Dict[str, Any]:
       
        client = MlflowClient(tracking_uri=self.tracking_uri)

        try:
            model_uri = f"runs:/{self.run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)

            if stage:
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage=stage
                )

            if description:
                client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )

            return {
                "name": model_name,
                "version": model_version.version,
                "stage": stage,
                "status": "registered"
            }

        except Exception as e:
            logger.error(f": {str(e)}")
            raise ModelTrainingException(
                message=f": {str(e)}",
                cause=e
            )

    def compare_runs(self, metric: str = "rmse") -> pd.DataFrame:
      
  
        client = MlflowClient(tracking_uri=self.tracking_uri)

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return pd.DataFrame()

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=10
        )

        results = []
        for run in runs:
            results.append({
                "run_id": run.info.run_id,
                "status": run.info.status,
                metric: run.data.metrics.get(metric, None),
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": run.data.params
            })

        return pd.DataFrame(results)


class ModelRegistry:
 
    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000"
        )
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def get_latest_model(
        self,
        name: str,
        stage: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
       
        try:
            if stage:
                model = self.client.get_latest_model_versions(name, stages=[stage])
            else:
                model = self.client.get_latest_model_versions(name)

            if not model:
                return None

            mv = model[0]
            return {
                "name": mv.name,
                "version": mv.version,
                "stage": mv.current_stage,
                "status": mv.status,
                "run_id": mv.run_id,
                "artifact_uri": mv.source,
                "created_at": mv.creation_timestamp
            }

        except Exception as e:
            logger.warning(f"获取模型失败: {str(e)}")
            return None

    def load_production_model(self, name: str) -> Any:
       
        import tempfile
        import shutil

        latest = self.get_latest_model(name, stage="Production")
        if not latest:
            raise ValueError(f": {name}")

        local_dir = tempfile.mkdtemp()
        model_path = self.client.download_artifacts(
            latest["run_id"],
            "model",
            local_dir
        )

        model = joblib.load(model_path)
        shutil.rmtree(local_dir)

        return model

    def list_models(self) -> List[Dict[str, Any]]:
        
        models = self.client.search_registered_models()
        return [
            {
                "name": m.name,
                "latest_version": m.latest_versions[0].version if m.latest_versions else None,
                "description": m.description,
                "tags": m.tags,
                "created_at": m.creation_timestamp
            }
            for m in models
        ]
