

import time
import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from functools import lru_cache
import json
import hashlib


class StructuredLogger:
    
    def __init__(
        self,
        name: str = "ml_pipeline",
        log_dir: str = "logs",
        log_level: str = "INFO"
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(
                log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] [%(request_id)s] %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _format_extra(self, extra: Dict[str, Any]) -> str:
   
        if not extra:
            return ""
        return f" | {json.dumps(extra, default=str)}"

    def _log(
        self,
        level: str,
        message: str,
        request_id: Optional[str] = None,
        **kwargs
    ):
      
        extra = kwargs.pop('extra', {})
        extra['request_id'] = request_id or 'no-request'

        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(f"{message}{self._format_extra(extra)}", extra=extra)

    def info(self, message: str, request_id: Optional[str] = None, **kwargs):
        self._log('INFO', message, request_id, **kwargs)

    def debug(self, message: str, request_id: Optional[str] = None, **kwargs):
        self._log('DEBUG', message, request_id, **kwargs)

    def warning(self, message: str, request_id: Optional[str] = None, **kwargs):
        self._log('WARNING', message, request_id, **kwargs)

    def error(self, message: str, request_id: Optional[str] = None, **kwargs):
        self._log('ERROR', message, request_id, **kwargs)

    def critical(self, message: str, request_id: Optional[str] = None, **kwargs):
        self._log('CRITICAL', message, request_id, **kwargs)

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        request_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_size: int = 0,
        response_size: int = 0,
        extra: Optional[Dict[str, Any]] = None
    ):
      
        log_data = {
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "client_ip": client_ip,
            "request_size": request_size,
            "response_size": response_size,
        }

        if extra:
            log_data.update(extra)

        if status_code >= 500:
            self.error(
                f"HTTP {method} {path} - {status_code}",
                request_id=request_id,
                extra=log_data
            )
        elif status_code >= 400:
            self.warning(
                f"HTTP {method} {path} - {status_code}",
                request_id=request_id,
                extra=log_data
            )
        else:
            self.info(
                f"HTTP {method} {path} - {status_code}",
                request_id=request_id,
                extra=log_data
            )

    def log_prediction(
        self,
        prediction: float,
        model_version: str,
        inference_time_ms: float,
        request_id: Optional[str] = None,
        features_hash: Optional[str] = None,
        status: str = "success",
        error: Optional[str] = None
    ):
       
        log_data = {
            "prediction": prediction,
            "model_version": model_version,
            "inference_time_ms": round(inference_time_ms, 2),
            "features_hash": features_hash,
            "status": status
        }

        if error:
            self.error(
                f"Prediction failed: {error}",
                request_id=request_id,
                extra=log_data
            )
        else:
            self.info(
                f"Prediction: {prediction}",
                request_id=request_id,
                extra=log_data
            )

    def log_training(
        self,
        run_id: str,
        status: str,
        duration_seconds: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        request_id: Optional[str] = None
    ):
       
        log_data = {
            "run_id": run_id,
            "status": status,
            "duration_seconds": round(duration_seconds, 2) if duration_seconds else None,
            "metrics": metrics
        }

        if status == "failed":
            self.error(
                f"Training failed: {run_id}",
                request_id=request_id,
                extra=log_data
            )
        else:
            self.info(
                f"Training {status}: {run_id}",
                request_id=request_id,
                extra=log_data
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
 

    def __init__(
        self,
        app,
        log_dir: str = "logs",
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.logger = StructuredLogger(log_dir=log_dir)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]

    def _should_log(self, path: str) -> bool:
   
        for exclude in self.exclude_paths:
            if path.startswith(exclude):
                return False
        return True

    def _get_client_ip(self, request: Request) -> str:

        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _generate_request_id(self) -> str:
        
        return str(uuid.uuid4())[:16]

    def _hash_features(self, data: bytes) -> str:
        
        return hashlib.md5(data).hexdigest()[:16]

    async def dispatch(self, request: Request, call_next) -> Response:
       
        request_id = request.headers.get("X-Request-ID") or self._generate_request_id()
        request.state.request_id = request_id

        if not self._should_log(request.path):
            return await call_next(request)

        start_time = time.time()
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")

        try:
            body = b""
            async for chunk in request.stream():
                body += chunk

            if body:
                request._body = body

            response = await call_next(request)

        except Exception as e:
            self.logger.error(
                f"Request failed: {str(e)}",
                request_id=request_id,
                extra={
                    "method": request.method,
                    "path": str(request.url.path),
                    "client_ip": client_ip
                }
            )
            raise

        duration_ms = (time.time() - start_time) * 1000

        self.logger.log_request(
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
            client_ip=client_ip,
            user_agent=user_agent,
            request_size=len(body) if body else 0,
            response_size=response.headers.get("content-length", 0)
        )

        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
   
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger("ml_pipeline.middleware")

    async def dispatch(self, request: Request, call_next):
       
        start_time = time.time()

        self.logger.debug(f"Request started: {request.method} {request.url.path}")

        response = await call_next(request)

        duration = time.time() - start_time
        self.logger.info(
            f"Request completed: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Duration: {duration:.3f}s"
        )

        return response


class AuditLogger:
    

    def __init__(self, log_dir: str = "logs"):
        self.audit_path = Path(log_dir) / "audit"
        self.audit_path.mkdir(parents=True, exist_ok=True)
        self.current_date = datetime.now().strftime("%Y%m%d")
        self._init_file()

    def _init_file(self):
        """初始化审计日志文件"""
        log_file = self.audit_path / f"audit_{self.current_date}.jsonl"
        if not log_file.exists():
            log_file.touch()

    def _rotate_if_needed(self):
        
        current = datetime.now().strftime("%Y%m%d")
        if current != self.current_date:
            self.current_date = current
            self._init_file()

    def log(
        self,
        action: str,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
       
        self._rotate_if_needed()

        log_file = self.audit_path / f"audit_{self.current_date}.jsonl"

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "status": status,
            "details": details or {},
            "request_id": request_id,
            "ip_address": ip_address
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + "\n")


class MetricsCollector:
   

    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, list] = defaultdict(list)
        self._gauges: Dict[str, float] = {}

    def increment(self, name: str, value: int = 1):
       
        self._counters[name] += value

    def record(self, name: str, value: float):
       
        self._histograms[name].append(value)
        if len(self._histograms[name]) > 1000:
            self._histograms[name] = self._histograms[name][-1000:]

    def gauge(self, name: str, value: float):
        
        self._gauges[name] = value

    def get_counter(self, name: str) -> int:
     
        return self._counters.get(name, 0)

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        
        values = self._histograms.get(name, [])
        if not values:
            return {}

        sorted_values = sorted(values)
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "p50": sorted_values[len(sorted_values) // 2],
            "p95": sorted_values[int(len(sorted_values) * 0.95)],
            "p99": sorted_values[int(len(sorted_values) * 0.99)]
        }

    def get_all_metrics(self) -> Dict[str, Any]:
     
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: self.get_histogram_stats(name)
                for name in self._histograms.keys()
            }
        }


@lru_cache()
def get_logger(name: str = "ml_pipeline") -> StructuredLogger:
   
    return StructuredLogger(name=name)


@lru_cache()
def get_audit_logger() -> AuditLogger:
  
    return AuditLogger()


@lru_cache()
def get_metrics_collector() -> MetricsCollector:
    """获取指标收集器单例"""
    return MetricsCollector()
