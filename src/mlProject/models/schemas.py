

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import numpy as np


class ModelType(str, Enum):
   
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class TrainingStatus(str, Enum):
   
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WineQualityInput(BaseModel):
   
    fixed_acidity: float = Field(
        ..., ge=0, le=20,
        description=" (g/L)",
        examples=[7.4, 7.8, 8.1]
    )
    volatile_acidity: float = Field(
        ..., ge=0, le=2,
        description=" (g/L)",
        examples=[0.7, 0.88, 0.5]
    )
    citric_acid: float = Field(
        ..., ge=0, le=2,
        description="(g/L)",
        examples=[0.0, 0.68, 0.56]
    )
    residual_sugar: float = Field(
        ..., ge=0, le=20,
        description=" (g/L)",
        examples=[1.9, 2.6, 6.9]
    )
    chlorides: float = Field(
        ..., ge=0, le=1,
        description="(g/L)",
        examples=[0.076, 0.092, 0.045]
    )
    free_sulfur_dioxide: float = Field(
        ..., ge=0, le=100,
        description=" (mg/L)",
        examples=[11, 25, 17]
    )
    total_sulfur_dioxide: float = Field(
        ..., ge=0, le=300,
        description=" (mg/L)",
        examples=[34, 67, 102]
    )
    density: float = Field(
        ..., ge=0.9, le=1.1,
        description="(g/cm³)",
        examples=[0.9978, 0.9968, 0.9976]
    )
    pH: float = Field(
        ..., ge=0, le=14,
        description="",
        examples=[3.51, 3.2, 3.26]
    )
    sulphates: float = Field(
        ..., ge=0, le=2,
        description="(g/L)",
        examples=[0.56, 0.65, 0.47]
    )
    alcohol: float = Field(
        ..., ge=8, le=15,
        description=" (%)",
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
    

    data: List[WineQualityInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=""
    )
    model_version: Optional[str] = Field(
        None,
        description=""
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
    
    prediction: float = Field(
        ...,
        description=""
    )
    confidence: Optional[float] = Field(
        None,
        ge=0, le=1,
        description=""
    )
    model_version: str = Field(
        ...,
        description=""
    )
    model_type: str = Field(
        ...,
        description=""
    )
    inference_time_ms: float = Field(
        ...,
        description=""
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description=""
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
    

    predictions: List[float] = Field(
        ...,
        description=""
    )
    count: int = Field(
        ...,
        description=""
    )
    model_version: str = Field(
        ...,
        description=""
    )
    total_inference_time_ms: float = Field(
        ...,
        description=""
    )
    avg_inference_time_ms: float = Field(
        ...,
        description=""
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description=""
    )


class TrainingRequest(BaseModel):
    

    model_type: ModelType = Field(
        default=ModelType.ELASTIC_NET,
        description=""
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        None,
        description=""
    )
    cross_validation_folds: int = Field(
        default=5,
        ge=2, le=10,
        description=""
    )
    enable_early_stopping: bool = Field(
        default=True,
        description=""
    )
    run_name: Optional[str] = Field(
        None,
        description=""
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
   

    run_id: str = Field(
        ...,
        description=""
    )
    status: TrainingStatus = Field(
        ...,
        description=""
    )
    metrics: Optional[Dict[str, float]] = Field(
        None,
        description=""
    )
    model_version: Optional[str] = Field(
        None,
        description=""
    )
    artifact_uri: Optional[str] = Field(
        None,
        description=""
    )
    start_time: datetime = Field(
        ...,
        description=""
    )
    end_time: Optional[datetime] = Field(
        None,
        description=""
    )
    duration_seconds: Optional[float] = Field(
        None,
        description=")"
    )


class ModelInfo(BaseModel):
  

    name: str = Field(..., description="")
    version: str = Field(..., description="")
    model_type: str = Field(..., description="")
    stage: str = Field(..., description="")
    status: str = Field(..., description="")
    created_at: datetime = Field(..., description="")
    last_updated: datetime = Field(..., description="")
    description: Optional[str] = Field(None, description="")
    tags: Optional[Dict[str, str]] = Field(None, description="")
    metrics: Optional[Dict[str, float]] = Field(None, description="")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="")


class HealthCheckResponse(BaseModel):
  

    status: str = Field(..., description="")
    version: str = Field(..., description="")
    uptime_seconds: float = Field(..., description="")
    dependencies: Dict[str, str] = Field(..., description="")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsResponse(BaseModel):
    

    total_requests: int = Field(..., description="")
    successful_predictions: int = Field(..., description="")
    failed_predictions: int = Field(..., description="")
    avg_inference_time_ms: float = Field(..., description="")
    model_versions_used: List[str] = Field(..., description="")
    requests_by_model_version: Dict[str, int] = Field(..., description="")


class ErrorResponse(BaseModel):
   

    error_code: int = Field(..., description="")
    error_name: str = Field(..., description="")
    message: str = Field(..., description="")
    details: Optional[Dict[str, Any]] = Field(None, description="")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="")
    request_id: Optional[str] = Field(None, description="")
    path: Optional[str] = Field(None, description="")
    method: Optional[str] = Field(None, description="")
    retryable: bool = Field(False, description="")


class SuccessResponse(BaseModel):
    

    success: bool = Field(True, description="")
    message: str = Field(..., description="")
    data: Optional[Dict[str, Any]] = Field(None, description="")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
