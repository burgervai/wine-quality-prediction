"""
Pydantic Models for API Request/Response Schemas
API请求/响应模式的数据模型
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import numpy as np


class ModelType(str, Enum):
    """支持的模型类型"""
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class TrainingStatus(str, Enum):
    """训练状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WineQualityInput(BaseModel):
    """红葡萄酒质量预测输入模型"""

    fixed_acidity: float = Field(
        ..., ge=0, le=20,
        description="固定酸度 (g/L)",
        examples=[7.4, 7.8, 8.1]
    )
    volatile_acidity: float = Field(
        ..., ge=0, le=2,
        description="挥发性酸度 (g/L)",
        examples=[0.7, 0.88, 0.5]
    )
    citric_acid: float = Field(
        ..., ge=0, le=2,
        description="柠檬酸 (g/L)",
        examples=[0.0, 0.68, 0.56]
    )
    residual_sugar: float = Field(
        ..., ge=0, le=20,
        description="残糖 (g/L)",
        examples=[1.9, 2.6, 6.9]
    )
    chlorides: float = Field(
        ..., ge=0, le=1,
        description="氯化物 (g/L)",
        examples=[0.076, 0.092, 0.045]
    )
    free_sulfur_dioxide: float = Field(
        ..., ge=0, le=100,
        description="游离二氧化硫 (mg/L)",
        examples=[11, 25, 17]
    )
    total_sulfur_dioxide: float = Field(
        ..., ge=0, le=300,
        description="总二氧化硫 (mg/L)",
        examples=[34, 67, 102]
    )
    density: float = Field(
        ..., ge=0.9, le=1.1,
        description="密度 (g/cm³)",
        examples=[0.9978, 0.9968, 0.9976]
    )
    pH: float = Field(
        ..., ge=0, le=14,
        description="pH值",
        examples=[3.51, 3.2, 3.26]
    )
    sulphates: float = Field(
        ..., ge=0, le=2,
        description="硫酸盐 (g/L)",
        examples=[0.56, 0.65, 0.47]
    )
    alcohol: float = Field(
        ..., ge=8, le=15,
        description="酒精度 (%)",
        examples=[9.4, 9.8, 10.3]
    )

    @field_validator('*', mode='before')
    @classmethod
    def validate_numeric(cls, v):
        if isinstance(v, str):
            return float(v)
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class BatchPredictionInput(BaseModel):
    """批量预测输入模型"""

    data: List[WineQualityInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="预测数据列表"
    )
    model_version: Optional[str] = Field(
        None,
        description="指定模型版本 (留空使用最新模型)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "data": [
                    {
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
                ],
                "model_version": "v1.0.0"
            }
        }
    }


class PredictionResponse(BaseModel):
    """预测响应模型"""

    prediction: float = Field(
        ...,
        description="预测的酒质量等级"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0, le=1,
        description="预测置信度"
    )
    model_version: str = Field(
        ...,
        description="模型版本"
    )
    model_type: str = Field(
        ...,
        description="模型类型"
    )
    inference_time_ms: float = Field(
        ...,
        description="推理耗时 (毫秒)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="预测时间戳"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 5.5,
                "confidence": 0.85,
                "model_version": "v1.0.0",
                "model_type": "elastic_net",
                "inference_time_ms": 12.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """批量预测响应模型"""

    predictions: List[float] = Field(
        ...,
        description="批量预测结果"
    )
    count: int = Field(
        ...,
        description="预测数量"
    )
    model_version: str = Field(
        ...,
        description="模型版本"
    )
    total_inference_time_ms: float = Field(
        ...,
        description="总推理耗时 (毫秒)"
    )
    avg_inference_time_ms: float = Field(
        ...,
        description="平均推理耗时 (毫秒)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="预测时间戳"
    )


class TrainingRequest(BaseModel):
    """训练请求模型"""

    model_type: ModelType = Field(
        default=ModelType.ELASTIC_NET,
        description="模型类型"
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        None,
        description="超参数配置"
    )
    cross_validation_folds: int = Field(
        default=5,
        ge=2, le=10,
        description="交叉验证折数"
    )
    enable_early_stopping: bool = Field(
        default=True,
        description="启用早停"
    )
    run_name: Optional[str] = Field(
        None,
        description="MLflow运行名称"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_type": "elastic_net",
                "hyperparameters": {
                    "alpha": 0.5,
                    "l1_ratio": 0.5
                },
                "cross_validation_folds": 5,
                "enable_early_stopping": True,
                "run_name": "production_run_v1"
            }
        }
    }


class TrainingResponse(BaseModel):
    """训练响应模型"""

    run_id: str = Field(
        ...,
        description="MLflow运行ID"
    )
    status: TrainingStatus = Field(
        ...,
        description="训练状态"
    )
    metrics: Optional[Dict[str, float]] = Field(
        None,
        description="训练指标"
    )
    model_version: Optional[str] = Field(
        None,
        description="模型版本"
    )
    artifact_uri: Optional[str] = Field(
        None,
        description="模型artifact URI"
    )
    start_time: datetime = Field(
        ...,
        description="开始时间"
    )
    end_time: Optional[datetime] = Field(
        None,
        description="结束时间"
    )
    duration_seconds: Optional[float] = Field(
        None,
        description="训练耗时 (秒)"
    )


class ModelInfo(BaseModel):
    """模型信息模型"""

    name: str = Field(..., description="模型名称")
    version: str = Field(..., description="模型版本")
    model_type: str = Field(..., description="模型类型")
    stage: str = Field(..., description="模型阶段")
    status: str = Field(..., description="模型状态")
    created_at: datetime = Field(..., description="创建时间")
    last_updated: datetime = Field(..., description="最后更新时间")
    description: Optional[str] = Field(None, description="模型描述")
    tags: Optional[Dict[str, str]] = Field(None, description="模型标签")
    metrics: Optional[Dict[str, float]] = Field(None, description="模型指标")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="输入模式")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="输出模式")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""

    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="API版本")
    uptime_seconds: float = Field(..., description="运行时间 (秒)")
    dependencies: Dict[str, str] = Field(..., description="依赖服务状态")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsResponse(BaseModel):
    """指标响应模型"""

    total_requests: int = Field(..., description="总请求数")
    successful_predictions: int = Field(..., description="成功预测数")
    failed_predictions: int = Field(..., description="失败预测数")
    avg_inference_time_ms: float = Field(..., description="平均推理时间 (毫秒)")
    model_versions_used: List[str] = Field(..., description="使用的模型版本")
    requests_by_model_version: Dict[str, int] = Field(..., description="按版本分的请求数")


class ErrorResponse(BaseModel):
    """错误响应模型"""

    error_code: int = Field(..., description="错误代码")
    error_name: str = Field(..., description="错误名称")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="错误时间戳")
    request_id: Optional[str] = Field(None, description="请求ID")
    path: Optional[str] = Field(None, description="请求路径")
    method: Optional[str] = Field(None, description="请求方法")
    retryable: bool = Field(False, description="是否可重试")


class SuccessResponse(BaseModel):
    """通用成功响应模型"""

    success: bool = Field(True, description="是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
