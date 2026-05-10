
import time
import hashlib
import hmac
import secrets
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps
from fastapi import HTTPException, Security, status, Request
from fastapi.security import APIKeyHeader, HTTPBearer, OAuth2PasswordBearer
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from pydantic import BaseModel


logger = logging.getLogger(__name__)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


class RateLimiter:

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'tokens': requests_per_minute,
            'last_update': time.time()
        })
        self.request_counts: Dict[str, list] = defaultdict(list)

    def _refill_bucket(self, client_id: str) -> None:
        bucket = self.buckets[client_id]
        now = time.time()
        elapsed = now - bucket['last_update']

        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        bucket['tokens'] = min(
            self.requests_per_minute,
            bucket['tokens'] + tokens_to_add
        )
        bucket['last_update'] = now

    def check_rate_limit(self, client_id: str, tokens: int = 1) -> tuple[bool, int]:
        """检查速率限制

        Returns:
            (allowed: bool, remaining_tokens: int)
        """
        self._refill_bucket(client_id)
        bucket = self.buckets[client_id]

        if bucket['tokens'] >= tokens:
            bucket['tokens'] -= tokens
            return True, int(bucket['tokens'])
        else:
            return False, 0

    def get_retry_after(self, client_id: str) -> int:
        """获取需要等待的秒数"""
        bucket = self.buckets[client_id]
        tokens_needed = 1
        if bucket['tokens'] < tokens_needed:
            tokens_shortage = tokens_needed - bucket['tokens']
            refill_rate = self.requests_per_minute / 60.0
            return int(tokens_shortage / refill_rate) + 1
        return 1

    def record_request(self, client_id: str) -> None:
        """记录请求"""
        now = time.time()
        self.request_counts[client_id].append(now)

        cutoff = now - 60
        self.request_counts[client_id] = [
            t for t in self.request_counts[client_id] if t > cutoff
        ]

    def get_request_count(self, client_id: str) -> int:
        """获取最近60秒内的请求数"""
        now = time.time()
        cutoff = now - 60
        return sum(1 for t in self.request_counts[client_id] if t > cutoff)


class TokenManager:
    """JWT令牌管理器"""

    def __init__(
        self,
        secret_key: str = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7
    ):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        self.access_token_expire = timedelta(minutes=access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=refresh_token_expire_days)
        self._blacklist = set()

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
        to_encode = data.copy()

        expire = datetime.utcnow() + (
            expires_delta or self.access_token_expire
        )
        to_encode.update({
            "exp": expire,
            "type": "access"
        })

        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(
        self,
        data: Dict[str, Any]
    ) -> str:
        """创建刷新令牌"""
        to_encode = data.copy()
        expire = datetime.utcnow() + self.refresh_token_expire
        to_encode.update({
            "exp": expire,
            "type": "refresh"
        })

        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Dict[str, Any]:
        """验证令牌"""
        try:
            if token in self._blacklist:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="令牌已被撤销"
                )

            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="令牌已过期"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的令牌"
            )

    def revoke_token(self, token: str) -> None:
        """撤销令牌"""
        self._blacklist.add(token)

    def get_user_from_token(self, token: str) -> Dict[str, Any]:
        """从令牌获取用户信息"""
        payload = self.verify_token(token)
        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", [])
        }


class APIKeyManager:
    """API密钥管理器"""

    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self._load_default_keys()

    def _load_default_keys(self):
        """加载默认API密钥"""
        import os
        default_key = os.environ.get("DEFAULT_API_KEY")
        if default_key:
            self.api_keys[default_key] = {
                "name": "default",
                "created_at": datetime.utcnow(),
                "rate_limit": 100,
                "permissions": ["predict", "train", "models"],
                "active": True
            }

    def create_api_key(
        self,
        name: str,
        rate_limit: int = 100,
        permissions: list = None
    ) -> tuple[str, str]:
        """创建新的API密钥"""
        key_id = secrets.token_urlsafe(16)
        secret = secrets.token_urlsafe(32)

        api_key = f"{key_id}.{secret}"
        key_hash = hashlib.sha256(secret.encode()).hexdigest()

        self.api_keys[key_hash] = {
            "name": name,
            "key_id": key_id,
            "created_at": datetime.utcnow(),
            "rate_limit": rate_limit,
            "permissions": permissions or ["predict"],
            "active": True
        }

        return api_key, key_id

    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """验证API密钥"""
        if not api_key:
            return None

        parts = api_key.split(".")
        if len(parts) != 2:
            return None

        key_id, secret = parts
        key_hash = hashlib.sha256(secret.encode()).hexdigest()

        key_info = self.api_keys.get(key_hash)
        if not key_info or not key_info.get("active"):
            return None

        return key_info

    def revoke_api_key(self, api_key: str) -> bool:
        """撤销API密钥"""
        parts = api_key.split(".")
        if len(parts) != 2:
            return False

        secret = parts[1]
        key_hash = hashlib.sha256(secret.encode()).hexdigest()

        if key_hash in self.api_keys:
            self.api_keys[key_hash]["active"] = False
            return True
        return False


# 全局实例
token_manager = TokenManager()
api_key_manager = APIKeyManager()
rate_limiter = RateLimiter()


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header)
) -> Dict[str, Any]:
    """验证API密钥依赖"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少API密钥",
            headers={"WWW-Authenticate": "API-Key"}
        )

    key_info = api_key_manager.verify_api_key(api_key)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API密钥",
            headers={"WWW-Authenticate": "API-Key"}
        )

    client_id = request.client.host if request.client else "unknown"
    allowed, remaining = rate_limiter.check_rate_limit(client_id)

    if not allowed:
        retry_after = rate_limiter.get_retry_after(client_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"速率限制超出，请 {retry_after} 秒后重试",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(key_info["rate_limit"]),
                "X-RateLimit-Remaining": str(remaining)
            }
        )

    rate_limiter.record_request(client_id)
    return key_info


async def verify_jwt_token(
    request: Request,
    token: Optional[str] = Security(bearer_scheme)
) -> Dict[str, Any]:
    """验证JWT令牌依赖"""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证令牌",
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        user_info = token_manager.get_user_from_token(token.credentials)
        return user_info
    except HTTPException as e:
        raise e


def require_permission(permission: str):
    """权限检查装饰器"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="无法获取请求对象"
                )

            api_key = request.headers.get("X-API-Key")
            key_info = api_key_manager.verify_api_key(api_key)

            if not key_info:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="无效的API密钥"
                )

            if permission not in key_info.get("permissions", []):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"缺少必要权限: {permission}"
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def require_role(role: str):
    """角色检查装饰器"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user_info = kwargs.get('user_info')
            if not user_info:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="未认证"
                )

            if role not in user_info.get("roles", []):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"缺少必要角色: {role}"
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


class RateLimitMiddleware(BaseHTTPMiddleware):
    """速率限制中间件"""

    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.rate_limiter = RateLimiter(requests_per_minute)

    async def dispatch(self, request: Request, call_next):
        client_id = request.client.host if request.client else "unknown"

        allowed, remaining = self.rate_limiter.check_rate_limit(client_id)

        if not allowed:
            retry_after = self.rate_limiter.get_retry_after(client_id)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"速率限制超出，请 {retry_after} 秒后重试",
                    "retry_after": retry_after,
                    "remaining": remaining
                },
                headers={"Retry-After": str(retry_after)}
            )

        self.rate_limiter.record_request(client_id)
        response = await call_next(request)

        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.requests_per_minute)

        return response


from fastapi.responses import JSONResponse
