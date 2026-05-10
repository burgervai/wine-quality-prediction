
import traceback
import functools
import time
import logging
from typing import Callable, Any, Optional, Dict, Type, Tuple
from datetime import datetime
from enum import Enum


class ErrorCode(Enum):
    DATA_INGESTION_ERROR = 1001
    DATA_VALIDATION_ERROR = 1002
    DATA_TRANSFORMATION_ERROR = 1003
    DATA_NOT_FOUND = 1004
    DATA_FORMAT_ERROR = 1005
    DATA_SCHEMA_MISMATCH = 1006

    MODEL_TRAINING_ERROR = 2001
    MODEL_NOT_FOUND = 2002
    MODEL_LOADING_ERROR = 2003
    MODEL_PREDICTION_ERROR = 2004
    MODEL_EVALUATION_ERROR = 2005
    HYPERPARAMETER_ERROR = 2006

    API_REQUEST_ERROR = 3001
    API_RESPONSE_ERROR = 3002
    API_AUTHENTICATION_ERROR = 3003
    API_AUTHORIZATION_ERROR = 3004
    API_RATE_LIMIT_ERROR = 3005
    API_VALIDATION_ERROR = 3006
    API_TIMEOUT_ERROR = 3007

    CONFIG_ERROR = 4001
    CONFIG_NOT_FOUND = 4002
    CONFIG_VALIDATION_ERROR = 4003
    CONFIG_ENV_VAR_MISSING = 4004

    SYSTEM_ERROR = 5001
    DATABASE_ERROR = 5002
    CACHE_ERROR = 5003
    FILE_SYSTEM_ERROR = 5004
    NETWORK_ERROR = 5005

 
    UNKNOWN_ERROR = 9001
    VALIDATION_ERROR = 9002
    TIMEOUT_ERROR = 9003
    RETRY_EXHAUSTED = 9004


class BaseProductionException(Exception):
  

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        retryable: bool = False,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.retryable = retryable
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        self.request_id = self.context.get('request_id', 'unknown')
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        
        return {
            'error_code': self.error_code.value,
            'error_name': self.error_code.name,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'request_id': self.request_id,
            'retryable': self.retryable,
            'cause': str(self.cause) if self.cause else None,
            'traceback': traceback.format_exc() if self.cause else None
        }

    def __str__(self) -> str:
        return f"[{self.error_code.name}] {self.message}"


class DataIngestionException(BaseProductionException):


    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.DATA_INGESTION_ERROR,
            details=details,
            cause=cause,
            retryable=True,
            context=context
        )


class DataValidationException(BaseProductionException):
  

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.DATA_VALIDATION_ERROR,
            details=details,
            cause=cause,
            retryable=False,
            context=context
        )


class DataTransformationException(BaseProductionException):
  

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.DATA_TRANSFORMATION_ERROR,
            details=details,
            cause=cause,
            retryable=True,
            context=context
        )


class ModelTrainingException(BaseProductionException):
   

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_TRAINING_ERROR,
            details=details,
            cause=cause,
            retryable=True,
            context=context
        )


class ModelNotFoundException(BaseProductionException):
   

    def __init__(
        self,
        message: str,
        model_path: Optional[str] = None,
        model_version: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = {'model_path': model_path, 'model_version': model_version}
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_NOT_FOUND,
            details=details,
            retryable=False,
            context=context
        )


class ModelPredictionException(BaseProductionException):
 
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_PREDICTION_ERROR,
            details=details,
            cause=cause,
            retryable=True,
            context=context
        )


class ConfigurationException(BaseProductionException):
   
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if config_key:
            details['config_key'] = config_key
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIG_ERROR,
            details=details,
            cause=cause,
            retryable=False,
            context=context
        )


class APIException(BaseProductionException):
    

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details['status_code'] = status_code
        super().__init__(
            message=message,
            error_code=ErrorCode.API_REQUEST_ERROR,
            details=details,
            cause=cause,
            retryable=status_code >= 500,
            context=context
        )


class RateLimitException(APIException):


    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        details = {'retry_after': retry_after}
        super().__init__(
            message=message,
            status_code=429,
            details=details,
            context=context
        )
        self.error_code = ErrorCode.API_RATE_LIMIT_ERROR
        self.retryable = True


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        break

                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    if jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)

                    time.sleep(delay)

                    logging.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {str(e)}"
                    )

            raise last_exception

        return wrapper
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
) -> Callable:
    

    def decorator(func: Callable) -> Callable:
        failure_count = 0
        last_failure_time = None
        circuit_open = False

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nonlocal failure_count, last_failure_time, circuit_open

            if circuit_open:
                time_since_failure = time.time() - last_failure_time
                if time_since_failure < recovery_timeout:
                    raise APIException(
                        message="Circuit breaker is open",
                        details={
                            'failure_count': failure_count,
                            'recovery_timeout': recovery_timeout,
                            'time_since_failure': time_since_failure
                        }
                    )
                else:
                    circuit_open = False
                    failure_count = 0

            try:
                result = func(*args, **kwargs)
                if circuit_open:
                    logging.info(f"Circuit breaker closed for {func.__name__}")
                circuit_open = False
                failure_count = 0
                return result

            except expected_exception as e:
                failure_count += 1
                last_failure_time = time.time()

                if failure_count >= failure_threshold:
                    circuit_open = True
                    logging.error(
                        f"Circuit breaker opened for {func.__name__} "
                        f"after {failure_count} failures"
                    )

                raise e

        return wrapper
    return decorator


def handle_exceptions(
    default_return: Any = None,
    log_traceback: bool = True,
    custom_handler: Optional[Callable[[Exception], Any]] = None
) -> Callable:
    

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except BaseProductionException as e:
                logging.error(f"Production exception in {func.__name__}: {e}")
                if custom_handler:
                    return custom_handler(e)
                return default_return
            except Exception as e:
                logging.error(
                    f"Unexpected exception in {func.__name__}: {str(e)}",
                    exc_info=log_traceback
                )
                if custom_handler:
                    return custom_handler(e)
                return default_return

        return wrapper
    return decorator


class ExceptionHandler:
   

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: list = []
        self.max_history_size = 1000

    def handle(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        

        if isinstance(exception, BaseProductionException):
            error_response = exception.to_dict()
        else:
            error_response = {
                'error_code': ErrorCode.UNKNOWN_ERROR.value,
                'error_name': ErrorCode.UNKNOWN_ERROR.name,
                'message': str(exception),
                'timestamp': datetime.utcnow().isoformat(),
                'retryable': True,
                'cause': None,
                'traceback': traceback.format_exc()
            }

        error_response['context'] = context or {}

        self._record_error(error_response)
        self.logger.error(f"Exception handled: {error_response}")

        return error_response

    def _record_error(self, error: Dict[str, Any]) -> None:
        "
        self.error_history.append(error)
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]

    def get_error_summary(self) -> Dict[str, Any]:
      
        if not self.error_history:
            return {'total_errors': 0, 'error_types': {}}

        error_types = {}
        for error in self.error_history:
            error_name = error.get('error_name', 'UNKNOWN')
            error_types[error_name] = error_types.get(error_name, 0) + 1

        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'last_error': self.error_history[-1] if self.error_history else None,
            'retryable_count': sum(1 for e in self.error_history if e.get('retryable'))
        }
