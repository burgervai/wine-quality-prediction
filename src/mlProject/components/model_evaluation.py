
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
            # 加载数据和模型
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)

            # 准备数据
            test_x = test_data.drop([self.config.target_column], axis=1)
            test_y = test_data[[self.config.target_column]]

            # 计算指标
            predictions = model.predict(test_x)
            metrics = self._calculate_metrics(test_y, predictions)

            # 保存本地评估结果
            self._save_metrics(metrics)

            # 记录到MLflow
            self._log_to_mlflow(model, predictions, metrics)

            self.evaluation_result = {
                "status": "success",
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"模型评估完成: {metrics}")
            return self.evaluation_result

        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            raise ModelEvaluationException(
                message=f"模型评估失败: {str(e)}",
                cause=e
            )

    def _calculate_metrics(
        self,
        y_true: pd.DataFrame,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测值

        Returns:
            指标字典
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 计算额外指标
        mape = self._calculate_mape(y_true.values.flatten(), y_pred)

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "sample_count": len(y_true)
        }

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算平均绝对百分比误差

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            MAPE值
        """
        y_true = np.array(y_true)
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def _save_metrics(self, metrics: Dict[str, float]) -> None:
        """保存指标到本地文件"""
        os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)
        save_json(
            path=Path(self.config.metric_file_name),
            data=metrics
        )
        logger.info(f"指标已保存: {self.config.metric_file_name}")

    def _log_to_mlflow(
        self,
        model,
        predictions: np.ndarray,
        metrics: Dict[str, float]
    ) -> None:
        """记录到MLflow"""
        try:
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run(run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # 记录参数
                if hasattr(self.config, 'all_params'):
                    mlflow.log_params(self.config.all_params)

                # 记录指标
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # 记录模型
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
        """
        与基线模型比较

        Args:
            baseline_metrics: 基线指标

        Returns:
            比较结果
        """
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
        """
        生成评估报告

        Returns:
            报告字符串
        """
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
        """计算指标"""
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        """记录到MLflow"""
        return self.evaluator.evaluate()


if __name__ == '__main__':
    from mlProject.config.configuration import ConfigurationManager

    config = ConfigurationManager()
    model_eval_config = config.get_model_evaluation_config()
    evaluator = ProductionModelEvaluator(model_eval_config)
    result = evaluator.evaluate()
    print(f"Evaluation Result: {result}")