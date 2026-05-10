
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import joblib
from pathlib import Path
import json

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from mlProject import logger
from mlProject.core.exceptions import ModelEvaluationException


class ProductionModelEvaluator:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.evaluation_result: Dict[str, Any] = {}

    def evaluate(self) -> Dict[str, Any]:
        try:
        
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)

        
            test_x = test_data.drop([self.config.target_column], axis=1)
            test_y = test_data[[self.config.target_column]]

      
            predictions = model.predict(test_x)
            metrics = self._calculate_metrics(test_y, predictions)

            
            self._save_metrics(metrics)

        
            self._log_to_mlflow(model, predictions, metrics)

            self.evaluation_result = {
                "status": "success",
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f": {metrics}")
            return self.evaluation_result

        except Exception as e:
            logger.error(f": {str(e)}")
            raise ModelEvaluationException(
                message=f": {str(e)}",
                cause=e
            )

    def _calculate_metrics(
        self,
        y_true: pd.DataFrame,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
       
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        mape = self._calculate_mape(y_true.values.flatten(), y_pred)

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "sample_count": len(y_true)
        }

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
       
        y_true = np.array(y_true)
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def _save_metrics(self, metrics: Dict[str, float]) -> None:

        os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)
        save_json(
            path=Path(self.config.metric_file_name),
            data=metrics
        )
        logger.info(f": {self.config.metric_file_name}")

    def _log_to_mlflow(
        self,
        model,
        predictions: np.ndarray,
        metrics: Dict[str, float]
    ) -> None:
     
        try:
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run(run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
          
                if hasattr(self.config, 'all_params'):
                    mlflow.log_params(self.config.all_params)

     
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                        registered_model_name=self.config.registered_model_name
                    )
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            logger.warning(f"MLflow记录失败: {str(e)}")

    def compare_with_baseline(
        self,
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
       
        current_metrics = self.evaluation_result.get("metrics", {})

        comparison = {}
        for metric_name in baseline_metrics:
            if metric_name in current_metrics:
                improvement = (
                    (baseline_metrics[metric_name] - current_metrics[metric_name])
                    / baseline_metrics[metric_name]
                    * 100
                )
                comparison[metric_name] = {
                    "current": current_metrics[metric_name],
                    "baseline": baseline_metrics[metric_name],
                    "improvement_percent": improvement
                }

        return comparison

    def generate_evaluation_report(self) -> str:
        
        metrics = self.evaluation_result.get("metrics", {})

        report = f"""
Model Evaluation Report
=======================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Metrics:
--------
RMSE:  {metrics.get('rmse', 'N/A'):.4f}
MAE:   {metrics.get('mae', 'N/A'):.4f}
R2:    {metrics.get('r2', 'N/A'):.4f}
MAPE:  {metrics.get('mape', 'N/A'):.2f}%

Sample Count: {metrics.get('sample_count', 'N/A')}
"""
        return report


class ModelEvaluation:
    """Legacy model evaluation class"""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.evaluator = ProductionModelEvaluator(config)

    def eval_metrics(self, actual, pred):
        
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):

        return self.evaluator.evaluate()


if __name__ == '__main__':
    from mlProject.config.configuration import ConfigurationManager

    config = ConfigurationManager()
    model_eval_config = config.get_model_evaluation_config()
    evaluator = ProductionModelEvaluator(model_eval_config)
    result = evaluator.evaluate()
    print(f"Evaluation Result: {result}")
