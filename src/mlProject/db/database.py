
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from pathlib import Path
import json

import sqlite3
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, Column, String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func, and_, or_

logger = logging.getLogger(__name__)

Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    input_features = Column(Text, nullable=False)
    prediction = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), default="elastic_net")
    inference_time_ms = Column(Float, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    request_id = Column(String(100), nullable=True)
    client_ip = Column(String(50), nullable=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(100), unique=True, nullable=False)
    run_name = Column(String(200), nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    model_type = Column(String(50), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    metrics_json = Column(Text, nullable=True)
    parameters_json = Column(Text, nullable=True)
    artifact_uri = Column(String(500), nullable=True)
    mlflow_experiment = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    stage = Column(String(20), default="None")
    status = Column(String(20), default="pending")
    description = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    tags_json = Column(Text, nullable=True)
    artifact_path = Column(String(500), nullable=True)
    current_stage = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100), nullable=True)


class DatabaseManager:

    def __init__(self, config):
        self.config = config
        self.engine = None
        self.async_session = None
        self._connected = False

    async def initialize(self) -> bool:
        try:
            if self.config.use_sqlite:
                db_path = Path(self.config.sqlite_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)

                db_url = f"sqlite+aiosqlite:///{self.config.sqlite_path}"
                self.engine = create_async_engine(
                    db_url,
                    echo=False,
                    pool_size=self.config.connection_pool_size,
                    pool_recycle=3600
                )
            else:
                db_url = (
                    f"postgresql+asyncpg://{self.config.user}:{self.config.password}"
                    f"@{self.config.host}:{self.config.port}/{self.config.database}"
                )
                self.engine = create_async_engine(
                    db_url,
                    echo=False,
                    pool_size=self.config.connection_pool_size,
                    pool_recycle=3600
                )

            self.async_session = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )

            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._connected = True
            logger.info("数据库连接初始化成功")
            return True

        except Exception as e:
            logger.error(f"数据库连接初始化失败: {str(e)}")
            self._connected = False
            return False

    async def close(self):
        if self.engine:
            await self.engine.dispose()
            self._connected = False
            logger.info("数据库连接已关闭")

    def is_connected(self) -> bool:
        return self._connected

    @asynccontextmanager
    async def session(self):
        if not self._connected:
            raise ConnectionError("数据库未连接")

        session = self.async_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


class PredictionRepository:

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def record_prediction(
        self,
        input_features: Optional[List[List[float]]],
        prediction: Optional[float],
        model_version: str,
        model_type: str = "elastic_net",
        inference_time_ms: float = 0.0,
        status: str = "success",
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
        client_ip: Optional[str] = None
    ) -> int:
        async with self.db_manager.session() as session:
            pred = Prediction(
                input_features=json.dumps(input_features) if input_features else None,
                prediction=prediction,
                model_version=model_version,
                model_type=model_type,
                inference_time_ms=inference_time_ms,
                status=status,
                error_message=error_message,
                request_id=request_id,
                client_ip=client_ip
            )
            session.add(pred)
            await session.flush()
            return pred.id

    async def get_prediction_stats(self) -> Dict[str, Any]:
        async with self.db_manager.session() as session:
            total = await session.scalar(select(func.count(Prediction.id)))

            successful = await session.scalar(
                select(func.count(Prediction.id)).where(
                    Prediction.status == "success"
                )
            )

            failed = await session.scalar(
                select(func.count(Prediction.id)).where(
                    Prediction.status == "failed"
                )
            )

            avg_time = await session.scalar(
                select(func.avg(Prediction.inference_time_ms)).where(
                    Prediction.status == "success"
                )
            )

            return {
                "total": total or 0,
                "successful": successful or 0,
                "failed": failed or 0,
                "avg_inference_time": avg_time or 0.0
            }

    async def get_predictions_by_model(
        self,
        model_version: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        async with self.db_manager.session() as session:
            results = await session.execute(
                select(Prediction)
                .where(Prediction.model_version == model_version)
                .order_by(Prediction.created_at.desc())
                .limit(limit)
            )
            predictions = results.scalars().all()

            return [
                {
                    "id": p.id,
                    "prediction": p.prediction,
                    "model_version": p.model_version,
                    "status": p.status,
                    "inference_time_ms": p.inference_time_ms,
                    "created_at": p.created_at.isoformat() if p.created_at else None
                }
                for p in predictions
            ]

    async def get_recent_predictions(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        async with self.db_manager.session() as session:
            results = await session.execute(
                select(Prediction)
                .where(Prediction.created_at >= cutoff)
                .order_by(Prediction.created_at.desc())
                .limit(limit)
            )
            predictions = results.scalars().all()

            return [
                {
                    "id": p.id,
                    "prediction": p.prediction,
                    "model_version": p.model_version,
                    "status": p.status,
                    "created_at": p.created_at.isoformat() if p.created_at else None
                }
                for p in predictions
            ]

    async def delete_old_predictions(self, days: int = 30) -> int:
        cutoff = datetime.utcnow() - timedelta(days=days)

        async with self.db_manager.session() as session:
            result = await session.execute(
                select(func.count(Prediction.id)).where(
                    Prediction.created_at < cutoff
                )
            )
            count = result.scalar()

            await session.execute(
                Prediction.__table__.delete().where(
                    Prediction.created_at < cutoff
                )
            )

            return count


class TrainingRepository:

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def record_training_run(
        self,
        run_id: str,
        model_type: str,
        start_time: datetime,
        run_name: Optional[str] = None,
        status: str = "running"
    ) -> int:
        async with self.db_manager.session() as session:
            run = TrainingRun(
                run_id=run_id,
                run_name=run_name,
                model_type=model_type,
                start_time=start_time,
                status=status
            )
            session.add(run)
            await session.flush()
            return run.id

    async def update_training_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        end_time: Optional[datetime] = None,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        artifact_uri: Optional[str] = None
    ) -> bool:
        async with self.db_manager.session() as session:
            result = await session.execute(
                select(TrainingRun).where(TrainingRun.run_id == run_id)
            )
            run = result.scalar_one_or_none()

            if not run:
                return False

            if status:
                run.status = status
            if end_time:
                run.end_time = end_time
                if run.start_time:
                    run.duration_seconds = (end_time - run.start_time).total_seconds()
            if metrics:
                run.metrics_json = json.dumps(metrics)
            if parameters:
                run.parameters_json = json.dumps(parameters)
            if artifact_uri:
                run.artifact_uri = artifact_uri

            return True

    async def get_training_stats(self) -> Dict[str, Any]:
        async with self.db_manager.session() as session:
            total_runs = await session.scalar(select(func.count(TrainingRun.id)))

            completed_runs = await session.scalar(
                select(func.count(TrainingRun.id)).where(
                    TrainingRun.status == "completed"
                )
            )

            failed_runs = await session.scalar(
                select(func.count(TrainingRun.id)).where(
                    TrainingRun.status == "failed"
                )
            )

            avg_duration = await session.scalar(
                select(func.avg(TrainingRun.duration_seconds)).where(
                    TrainingRun.status == "completed"
                )
            )

            recent_runs = await session.execute(
                select(TrainingRun)
                .order_by(TrainingRun.created_at.desc())
                .limit(10)
            )
            recent = recent_runs.scalars().all()

            return {
                "total_runs": total_runs or 0,
                "completed_runs": completed_runs or 0,
                "failed_runs": failed_runs or 0,
                "avg_duration_seconds": avg_duration or 0.0,
                "recent_runs": [
                    {
                        "run_id": r.run_id,
                        "status": r.status,
                        "duration_seconds": r.duration_seconds,
                        "created_at": r.created_at.isoformat() if r.created_at else None
                    }
                    for r in recent
                ]
            }


class ModelRepository:

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def register_model(
        self,
        name: str,
        version: str,
        model_type: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        artifact_path: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> int:
        async with self.db_manager.session() as session:
            await session.execute(
                ModelRegistry.__table__.update()
                .where(ModelRegistry.name == name)
                .values(current_stage=False)
            )

            model = ModelRegistry(
                name=name,
                version=version,
                model_type=model_type,
                description=description,
                tags_json=json.dumps(tags) if tags else None,
                metrics_json=json.dumps(metrics) if metrics else None,
                artifact_path=artifact_path,
                current_stage=True,
                created_by=created_by
            )
            session.add(model)
            await session.flush()
            return model.id

    async def get_model_versions(self, name: str) -> List[Dict[str, Any]]:
        async with self.db_manager.session() as session:
            results = await session.execute(
                select(ModelRegistry)
                .where(ModelRegistry.name == name)
                .order_by(ModelRegistry.created_at.desc())
            )
            models = results.scalars().all()

            return [
                {
                    "name": m.name,
                    "version": m.version,
                    "model_type": m.model_type,
                    "stage": m.stage,
                    "status": m.status,
                    "current_stage": m.current_stage,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "last_updated": m.last_updated.isoformat() if m.last_updated else None
                }
                for m in models
            ]

    async def get_current_model(self, name: str) -> Optional[Dict[str, Any]]:
        async with self.db_manager.session() as session:
            result = await session.execute(
                select(ModelRegistry)
                .where(
                    and_(
                        ModelRegistry.name == name,
                        ModelRegistry.current_stage == True
                    )
                )
            )
            model = result.scalar_one_or_none()

            if not model:
                return None

            return {
                "name": model.name,
                "version": model.version,
                "model_type": model.model_type,
                "stage": model.stage,
                "artifact_path": model.artifact_path
            }
