
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow
from mlflow.tracking import MlflowClient

from mlProject.core.config import get_config_manager, APIConfig
from mlProject.core.exceptions import (
    APIException,
    ModelNotFoundException,
    ModelPredictionException,
    BaseProductionException
)
from mlProject.models.schemas import (
    WineQualityInput,
    PredictionResponse,
    BatchPredictionInput,
    BatchPredictionResponse,
    TrainingRequest,
    TrainingResponse,
    ModelInfo,
    HealthCheckResponse,
    ErrorResponse,
    SuccessResponse,
    TrainingStatus
)
from mlProject.middleware.auth import (
    verify_api_key,
    verify_jwt_token,
    RateLimiter
)
from mlProject.middleware.logging import LoggingMiddleware, RequestLoggingMiddleware
from mlProject.db.database import DatabaseManager
from mlProject.db.repositories import PredictionRepository, TrainingRepository


class AppState:

    def __init__(self):
        self.config_manager = None
        self.model = None
        self.model_version = None
        self.model_loaded_at = None
        self.start_time = datetime.utcnow()
        self.db_manager = None
        self.mlflow_client = None
        self.rate_limiter = RateLimiter(requests_per_minute=100)

    def reload_model(self) -> bool:
        try:
            config = self.config_manager.get_model_trainer_config()
            model_path = Path(config.root_dir) / config.model_name

            if not model_path.exists():
                raise ModelNotFoundException(
                    message="模型文件不存在",
                    model_path=str(model_path)
                )

            self.model = joblib.load(model_path)
            self.model_version = f"v{int(time.time())}"
            self.model_loaded_at = datetime.utcnow()

            logging.info(f"模型已重新加载: {self.model_version}")
            return True

        except Exception as e:
            logging.error(f"模型重新加载失败: {str(e)}")
            return False

    def get_uptime_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("正在启动生产级ML管道API服务...")

    app_state.config_manager = get_config_manager()

    db_config = app_state.config_manager.get_database_config()
    app_state.db_manager = DatabaseManager(db_config)
    await app_state.db_manager.initialize()

    eval_config = app_state.config_manager.get_model_evaluation_config()
    mlflow.set_tracking_uri(eval_config.mlflow_uri)
    app_state.mlflow_client = MlflowClient()

    app_state.reload_model()

    logging.info("API服务启动完成")

    yield

    logging.info("正在关闭API服务...")
    if app_state.db_manager:
        await app_state.db_manager.close()
    logging.info("API服务已关闭")


def create_app() -> FastAPI:

    api_config = app_state.config_manager.get_api_config() if app_state.config_manager else None

    app = FastAPI(
        title="ML Pipeline API",
        description="生产级机器学习管道API服务",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    register_routes(app)

    @app.exception_handler(BaseProductionException)
    async def production_exception_handler(request: Request, exc: BaseProductionException):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code=exc.error_code.value,
                error_name=exc.error_code.name,
                message=exc.message,
                details=exc.details,
                request_id=request.state.request_id if hasattr(request.state, 'request_id') else None,
                path=str(request.url.path),
                method=request.method,
                retryable=exc.retryable
            ).model_dump()
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error_code=exc.status_code,
                error_name="HTTP_ERROR",
                message=exc.detail,
                path=str(request.url.path),
                method=request.method
            ).model_dump()
        )

    return app


def register_routes(app: FastAPI):
    """注册所有路由"""

    @app.get("/", response_model=SuccessResponse)
    async def root():
        """API根路径"""
        return SuccessResponse(
            message="ML Pipeline API服务正在运行",
            data={
                "version": "2.0.0",
                "docs": "/docs",
                "health": "/health"
            }
        )

    @app.get("/health", response_model=HealthCheckResponse, tags=["System"])
    async def health_check(request: Request):
        """健康检查端点"""
        dependencies = {
            "model": "healthy" if app_state.model else "unhealthy",
            "database": "healthy" if app_state.db_manager else "unhealthy",
            "mlflow": "healthy" if app_state.mlflow_client else "unhealthy"
        }

        overall_status = "healthy" if all(
            v == "healthy" for v in dependencies.values()
        ) else "degraded"

        return HealthCheckResponse(
            status=overall_status,
            version="2.0.0",
            uptime_seconds=app_state.get_uptime_seconds(),
            dependencies=dependencies
        )

    @app.get("/metrics", tags=["System"])
    async def get_metrics(
        request: Request,
        api_key: str = Depends(verify_api_key)
    ):
        """获取系统指标"""
        if not app_state.db_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="数据库未连接"
            )

        pred_repo = PredictionRepository(app_state.db_manager)
        stats = await pred_repo.get_prediction_stats()

        return {
            "total_requests": stats.get("total", 0),
            "successful_predictions": stats.get("successful", 0),
            "failed_predictions": stats.get("failed", 0),
            "avg_inference_time_ms": stats.get("avg_inference_time", 0),
            "model_version": app_state.model_version,
            "uptime_seconds": app_state.get_uptime_seconds()
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
    async def predict(
        input_data: WineQualityInput,
        request: Request,
        api_key: str = Depends(verify_api_key)
    ):
        """单个预测"""
        start_time = time.time()

        if not app_state.model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="模型未加载"
            )

        try:
            data = np.array([[
                input_data.fixed_acidity,
                input_data.volatile_acidity,
                input_data.citric_acid,
                input_data.residual_sugar,
                input_data.chlorides,
                input_data.free_sulfur_dioxide,
                input_data.total_sulfur_dioxide,
                input_data.density,
                input_data.pH,
                input_data.sulphates,
                input_data.alcohol
            ]])

            prediction = float(app_state.model.predict(data)[0])
            inference_time = (time.time() - start_time) * 1000

            if app_state.db_manager:
                pred_repo = PredictionRepository(app_state.db_manager)
                await pred_repo.record_prediction(
                    input_features=data.tolist(),
                    prediction=prediction,
                    model_version=app_state.model_version,
                    inference_time_ms=inference_time,
                    status="success"
                )

            return PredictionResponse(
                prediction=prediction,
                confidence=None,
                model_version=app_state.model_version,
                model_type="elastic_net",
                inference_time_ms=inference_time
            )

        except Exception as e:
            inference_time = (time.time() - start_time) * 1000

            if app_state.db_manager:
                pred_repo = PredictionRepository(app_state.db_manager)
                await pred_repo.record_prediction(
                    input_features=None,
                    prediction=None,
                    model_version=app_state.model_version,
                    inference_time_ms=inference_time,
                    status="failed",
                    error_message=str(e)
                )

            raise ModelPredictionException(
                message=f"预测失败: {str(e)}",
                cause=e
            )

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
    async def batch_predict(
        input_data: BatchPredictionInput,
        request: Request,
        api_key: str = Depends(verify_api_key)
    ):
        """批量预测"""
        start_time = time.time()

        if not app_state.model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="模型未加载"
            )

        try:
            data = np.array([[
                d.fixed_acidity,
                d.volatile_acidity,
                d.citric_acid,
                d.residual_sugar,
                d.chlorides,
                d.free_sulfur_dioxide,
                d.total_sulfur_dioxide,
                d.density,
                d.pH,
                d.sulphates,
                d.alcohol
            ] for d in input_data.data])

            predictions = app_state.model.predict(data).tolist()
            total_time = (time.time() - start_time) * 1000

            return BatchPredictionResponse(
                predictions=predictions,
                count=len(predictions),
                model_version=app_state.model_version,
                total_inference_time_ms=total_time,
                avg_inference_time_ms=total_time / len(predictions) if predictions else 0
            )

        except Exception as e:
            raise ModelPredictionException(
                message=f"批量预测失败: {str(e)}",
                cause=e
            )

    @app.post("/train", response_model=TrainingResponse, tags=["Training"])
    async def train_model(
        training_request: TrainingRequest,
        background_tasks: BackgroundTasks,
        request: Request,
        api_key: str = Depends(verify_api_key)
    ):
        """触发模型训练"""
        from mlProject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
        from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

        start_time = datetime.utcnow()

        try:
                mlflow.set_experiment(training_request.run_name or "default_experiment")

            with mlflow.start_run(run_name=training_request.run_name) as run:
                data_validation = await background_tasks.add_task(
                    "数据验证"
                )

                data_transformation = await background_tasks.add_task(
                    "数据转换"
                )

                train_pipeline = ModelTrainerTrainingPipeline()
                train_pipeline.main()

                eval_pipeline = ModelEvaluationTrainingPipeline()
                eval_pipeline.log_into_mlflow()

            end_time = datetime.utcnow()

            return TrainingResponse(
                run_id=run.info.run_id,
                status=TrainingStatus.COMPLETED,
                metrics=None,
                model_version=f"v{run.info.run_id[:8]}",
                artifact_uri=run.info.artifact_uri,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds()
            )

        except Exception as e:
            raise ModelTrainingException(
                message=f"训练失败: {str(e)}",
                cause=e
            )

    @app.get("/models", response_model=list[ModelInfo], tags=["Models"])
    async def list_models(
        request: Request,
        api_key: str = Depends(verify_api_key)
    ):
        """列出所有模型版本"""
        try:
            eval_config = app_state.config_manager.get_model_evaluation_config()
            client = app_state.mlflow_client or MlflowClient()

            registered_models = client.search_registered_models()

            models = []
            for rm in registered_models:
                for mv in rm.latest_versions:
                    models.append(ModelInfo(
                        name=mv.name,
                        version=mv.version,
                        model_type=mv.model_type,
                        stage=mv.current_stage,
                        status=mv.status,
                        created_at=mv.creation_timestamp,
                        last_updated=mv.last_updated_timestamp,
                        description=rm.description,
                        tags=rm.tags,
                        metrics=None
                    ))

            return models

        except Exception as e:
            raise APIException(
                message=f"获取模型列表失败: {str(e)}",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @app.get("/models/{model_name}/{version}", response_model=ModelInfo, tags=["Models"])
    async def get_model_info(
        model_name: str,
        version: str,
        request: Request,
        api_key: str = Depends(verify_api_key)
    ):
        """获取特定模型信息"""
        try:
            client = app_state.mlflow_client or MlflowClient()
            model_versions = client.get_model_version(model_name, int(version))

            return ModelInfo(
                name=model_versions.name,
                version=str(model_versions.version),
                model_type=model_versions.model_type,
                stage=model_versions.current_stage,
                status=model_versions.status,
                created_at=model_versions.creation_timestamp,
                last_updated=model_versions.last_updated_timestamp
            )

        except Exception as e:
            raise ModelNotFoundException(
                message=f"模型未找到: {model_name}/{version}",
                model_path=f"{model_name}/{version}"
            )

    @app.post("/models/{model_name}/{version}/stage/{stage}", tags=["Models"])
    async def transition_model_stage(
        model_name: str,
        version: str,
        stage: str,
        request: Request,
        api_key: str = Depends(verify_api_key)
    ):
        """转换模型阶段 (如 Staging -> Production)"""
        try:
            client = app_state.mlflow_client or MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=int(version),
                stage=stage
            )

            return SuccessResponse(
                message=f"模型 {model_name}/{version} 已转换到 {stage}",
                data={"model_name": model_name, "version": version, "stage": stage}
            )

        except Exception as e:
            raise APIException(
                message=f"模型阶段转换失败: {str(e)}",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @app.post("/models/reload", tags=["Models"])
    async def reload_model(
        request: Request,
        api_key: str = Depends(verify_api_key)
    ):
        """重新加载模型"""
        success = app_state.reload_model()

        if success:
            return SuccessResponse(
                message="模型重新加载成功",
                data={"model_version": app_state.model_version}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="模型重新加载失败"
            )

    @app.delete("/predictions/cleanup", tags=["Admin"])
    async def cleanup_old_predictions(
        request: Request,
        api_key: str = Depends(verify_api_key),
        days: int = 30
    ):
        """清理旧预测记录"""
        if not app_state.db_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="数据库未连接"
            )

        pred_repo = PredictionRepository(app_state.db_manager)
        deleted_count = await pred_repo.delete_old_predictions(days)

        return SuccessResponse(
            message=f"已删除 {deleted_count} 条超过 {days} 天的预测记录",
            data={"deleted_count": deleted_count}
        )


app = create_app()


def run_server():
    """运行服务器"""
    import uvicorn

    config_manager = get_config_manager()
    api_config = config_manager.get_api_config()

    uvicorn.run(
        "mlProject.api.main:app",
        host=api_config.host,
        port=api_config.port,
        workers=api_config.workers,
        reload=api_config.reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
