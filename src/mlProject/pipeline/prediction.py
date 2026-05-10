
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from datetime import datetime

from mlProject import logger
from mlProject.core.exceptions import (
    ModelNotFoundException,
    PredictionException,
    handle_exception
)


class ProductionPredictionPipeline:

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        use_mlflow: bool = False
    ):
        self.model = None
        self.model_path = model_path
        self.model_name = model_name
        self.use_mlflow = use_mlflow
        self.metadata: Dict[str, Any] = {}
        self._load_model()

    def _load_model(self) -> None:
        try:
            if self.use_mlflow and self.model_name:
                self._load_from_mlflow()
            elif self.model_path:
                self._load_from_file()
            else:
                default_path = Path('artifacts/model_trainer/model.joblib')
                if default_path.exists():
                    self.model_path = str(default_path)
                    self._load_from_file()
                else:
                    raise ModelNotFoundException(
                        message="模型文件不存在",
                        model_path=self.model_path or str(default_path)
                    )

            logger.info(f"模型加载成功: {self.model_path}")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def _load_from_file(self) -> None:
        """从文件加载模型"""
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise ModelNotFoundException(
                message="模型文件不存在",
                model_path=str(model_path)
            )

        self.model = joblib.load(model_path)
        self.metadata = {
            "source": "file",
            "path": str(model_path),
            "loaded_at": datetime.now().isoformat()
        }

    def _load_from_mlflow(self) -> None:
        """从MLflow加载模型"""
        import mlflow

        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()

            latest_versions = client.get_latest_model_versions(
                self.model_name,
                stages=["Production"]
            )

            if not latest_versions:
                latest_versions = client.get_latest_model_versions(
                    self.model_name,
                    stages=["Staging"]
                )

            if not latest_versions:
                raise ModelNotFoundException(
                    message=f"MLflow中未找到模型: {self.model_name}",
                    model_path=f"mlflow://{self.model_name}"
                )

            latest = latest_versions[0]
            model_uri = latest.source

            self.model = mlflow.sklearn.load_model(model_uri)
            self.metadata = {
                "source": "mlflow",
                "name": self.model_name,
                "version": latest.version,
                "stage": latest.current_stage,
                "run_id": latest.run_id,
                "loaded_at": datetime.now().isoformat()
            }

            logger.info(f"从MLflow加载模型: {self.model_name} v{latest.version}")

        except Exception as e:
            logger.error(f"从MLflow加载模型失败: {str(e)}")
            raise

    def predict(self, data: Union[np.ndarray, pd.DataFrame, List]) -> np.ndarray:
        """
        单个预测

        Args:
            data: 输入数据

        Returns:
            预测结果
        """
        try:
            # 转换数据格式
            if isinstance(data, list):
                data = np.array(data)
            elif isinstance(data, pd.DataFrame):
                data = data.values

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # 预测
            start_time = datetime.now()
            prediction = self.model.predict(data)
            inference_time = (datetime.now() - start_time).total_seconds() * 1000

            result = {
                "prediction": float(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
                "inference_time_ms": round(inference_time, 2),
                "model_metadata": self.metadata
            }

            logger.debug(f"预测完成: {result}")
            return prediction

        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise PredictionException(
                message=f"预测失败: {str(e)}",
                details={"model_path": self.model_path}
            )

    def predict_batch(
        self,
        data: Union[np.ndarray, pd.DataFrame, List]
    ) -> Dict[str, Any]:
        """
        批量预测

        Args:
            data: 输入数据

        Returns:
            批量预测结果
        """
        try:
            # 转换数据格式
            if isinstance(data, list):
                data = np.array(data)
            elif isinstance(data, pd.DataFrame):
                data = data.values

            # 预测
            start_time = datetime.now()
            predictions = self.model.predict(data)
            inference_time = (datetime.now() - start_time).total_seconds() * 1000

            result = {
                "predictions": predictions.tolist(),
                "count": len(predictions),
                "inference_time_ms": round(inference_time, 2),
                "model_metadata": self.metadata
            }

            logger.info(f"批量预测完成: {len(predictions)} 条数据")
            return result

        except Exception as e:
            logger.error(f"批量预测失败: {str(e)}")
            raise PredictionException(
                message=f"批量预测失败: {str(e)}",
                details={"data_shape": str(data.shape) if hasattr(data, 'shape') else 'unknown'}
            )

    def validate_input(self, data: np.ndarray, expected_features: int = 11) -> bool:
        """
        验证输入数据

        Args:
            data: 输入数据
            expected_features: 期望的特征数量

        Returns:
            是否有效
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != expected_features:
            raise PredictionException(
                message=f"特征数量不匹配: 期望 {expected_features}, 实际 {data.shape[1]}",
                details={"expected": expected_features, "actual": data.shape[1]}
            )

        return True

    def reload(self) -> None:
        """重新加载模型"""
        logger.info("重新加载模型...")
        self._load_model()

    @property
    def model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "loaded": self.model is not None,
            "path": self.model_path,
            "name": self.model_name,
            "source": "mlflow" if self.use_mlflow else "file",
            "metadata": self.metadata
        }


class PredictionPipeline:
    """Legacy prediction pipeline for backward compatibility"""

    def __init__(self):
        self.pipeline = ProductionPredictionPipeline()

    def predict(self, data):
        """预测"""
        return self.pipeline.predict(data)


WINE_QUALITY_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]


def create_prediction_input(features: Dict[str, float]) -> np.ndarray:
    """
    从特征字典创建预测输入

    Args:
        features: 特征字典

    Returns:
        预测输入数组
    """
    return np.array([[features.get(f, 0.0) for f in WINE_QUALITY_FEATURES]])


if __name__ == '__main__':
    pipeline = ProductionPredictionPipeline()

    test_data = np.array([[7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]])
    prediction = pipeline.predict(test_data)
    print(f"Prediction: {prediction}")