"""
Redis utilities for Visionary AI backend services.
"""

import json
import pickle
from typing import Any, Optional, Union, Dict
from datetime import datetime, timedelta

import aioredis
from aioredis import Redis

from shared.config import get_settings


# Global Redis client
redis_client: Optional[Redis] = None


async def init_redis() -> Redis:
    """Initialize Redis connection."""
    global redis_client
    
    settings = get_settings()
    
    redis_client = aioredis.from_url(
        settings.redis.url,
        max_connections=settings.redis.max_connections,
        decode_responses=settings.redis.decode_responses,
        retry_on_timeout=True,
        socket_keepalive=True,
        socket_keepalive_options={},
    )
    
    # Test connection
    await redis_client.ping()
    return redis_client


async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()


async def get_redis_client() -> Redis:
    """Get Redis client instance."""
    global redis_client
    if redis_client is None:
        redis_client = await init_redis()
    return redis_client


class RedisCache:
    """Redis cache manager with serialization support."""
    
    def __init__(self, redis_client: Redis, key_prefix: str = "visionary"):
        self.redis = redis_client
        self.key_prefix = key_prefix
    
    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.key_prefix}:{key}"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        try:
            value = await self.redis.get(self._make_key(key))
            if value is None:
                return default
            
            # Try to deserialize JSON first, then pickle
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return pickle.loads(value)
        except Exception:
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize_json: bool = True
    ) -> bool:
        """Set value in cache."""
        try:
            # Serialize value
            if serialize_json:
                try:
                    serialized_value = json.dumps(value, default=str)
                except (TypeError, ValueError):
                    serialized_value = pickle.dumps(value)
            else:
                serialized_value = pickle.dumps(value)
            
            # Set with TTL if provided
            if ttl:
                return await self.redis.setex(
                    self._make_key(key),
                    ttl,
                    serialized_value
                )
            else:
                return await self.redis.set(
                    self._make_key(key),
                    serialized_value
                )
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            result = await self.redis.delete(self._make_key(key))
            return result > 0
        except Exception:
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return await self.redis.exists(self._make_key(key)) > 0
        except Exception:
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        try:
            return await self.redis.expire(self._make_key(key), ttl)
        except Exception:
            return False
    
    async def ttl(self, key: str) -> int:
        """Get TTL for key."""
        try:
            return await self.redis.ttl(self._make_key(key))
        except Exception:
            return -1
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment numeric value."""
        try:
            return await self.redis.incrby(self._make_key(key), amount)
        except Exception:
            return 0
    
    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement numeric value."""
        try:
            return await self.redis.decrby(self._make_key(key), amount)
        except Exception:
            return 0


class RedisLock:
    """Distributed lock using Redis."""
    
    def __init__(self, redis_client: Redis, key: str, timeout: int = 10):
        self.redis = redis_client
        self.key = f"lock:{key}"
        self.timeout = timeout
        self.identifier = None
    
    async def acquire(self) -> bool:
        """Acquire lock."""
        import uuid
        
        self.identifier = str(uuid.uuid4())
        
        # Try to acquire lock
        result = await self.redis.set(
            self.key,
            self.identifier,
            nx=True,  # Only set if key doesn't exist
            ex=self.timeout  # Set expiration
        )
        
        return result is not None
    
    async def release(self) -> bool:
        """Release lock."""
        if not self.identifier:
            return False
        
        # Lua script to atomically check and delete
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        try:
            result = await self.redis.eval(
                lua_script,
                1,
                self.key,
                self.identifier
            )
            return result == 1
        except Exception:
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        if await self.acquire():
            return self
        else:
            raise RuntimeError(f"Could not acquire lock: {self.key}")
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.release()


class RedisQueue:
    """Simple queue implementation using Redis lists."""
    
    def __init__(self, redis_client: Redis, queue_name: str):
        self.redis = redis_client
        self.queue_name = f"queue:{queue_name}"
    
    async def push(self, item: Any) -> int:
        """Push item to queue."""
        try:
            serialized_item = json.dumps(item, default=str)
            return await self.redis.lpush(self.queue_name, serialized_item)
        except Exception:
            return 0
    
    async def pop(self, timeout: int = 0) -> Optional[Any]:
        """Pop item from queue."""
        try:
            if timeout > 0:
                result = await self.redis.brpop(self.queue_name, timeout=timeout)
                if result:
                    _, item = result
                    return json.loads(item)
            else:
                item = await self.redis.rpop(self.queue_name)
                if item:
                    return json.loads(item)
            return None
        except Exception:
            return None
    
    async def size(self) -> int:
        """Get queue size."""
        try:
            return await self.redis.llen(self.queue_name)
        except Exception:
            return 0
    
    async def clear(self) -> bool:
        """Clear queue."""
        try:
            await self.redis.delete(self.queue_name)
            return True
        except Exception:
            return False


class RedisRateLimiter:
    """Rate limiter using Redis."""
    
    def __init__(self, redis_client: Redis, key: str, limit: int, window: int):
        self.redis = redis_client
        self.key = f"rate_limit:{key}"
        self.limit = limit
        self.window = window
    
    async def is_allowed(self) -> bool:
        """Check if request is allowed."""
        try:
            current_time = int(datetime.utcnow().timestamp())
            window_start = current_time - self.window
            
            # Remove old entries
            await self.redis.zremrangebyscore(
                self.key,
                0,
                window_start
            )
            
            # Count current requests
            current_count = await self.redis.zcard(self.key)
            
            if current_count < self.limit:
                # Add current request
                await self.redis.zadd(
                    self.key,
                    {str(current_time): current_time}
                )
                await self.redis.expire(self.key, self.window)
                return True
            
            return False
        except Exception:
            # Allow request if Redis is down
            return True
    
    async def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        try:
            current_time = int(datetime.utcnow().timestamp())
            window_start = current_time - self.window
            
            # Remove old entries
            await self.redis.zremrangebyscore(
                self.key,
                0,
                window_start
            )
            
            # Count current requests
            current_count = await self.redis.zcard(self.key)
            return max(0, self.limit - current_count)
        except Exception:
            return self.limit


class RedisSessionStore:
    """Session store using Redis."""
    
    def __init__(self, redis_client: Redis, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def _make_key(self, session_id: str) -> str:
        """Create session key."""
        return f"session:{session_id}"
    
    async def create_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Create new session."""
        try:
            serialized_data = json.dumps(data, default=str)
            return await self.redis.setex(
                self._make_key(session_id),
                self.ttl,
                serialized_data
            )
        except Exception:
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        try:
            data = await self.redis.get(self._make_key(session_id))
            if data:
                return json.loads(data)
            return None
        except Exception:
            return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data."""
        try:
            serialized_data = json.dumps(data, default=str)
            return await self.redis.setex(
                self._make_key(session_id),
                self.ttl,
                serialized_data
            )
        except Exception:
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        try:
            result = await self.redis.delete(self._make_key(session_id))
            return result > 0
        except Exception:
            return False
    
    async def extend_session(self, session_id: str) -> bool:
        """Extend session TTL."""
        try:
            return await self.redis.expire(self._make_key(session_id), self.ttl)
        except Exception:
            return False


# Health check function for Redis
async def check_redis_health() -> dict:
    """Check Redis connectivity."""
    try:
        redis = await get_redis_client()
        await redis.ping()
        return {"status": "healthy", "message": "Redis connection successful"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Redis connection failed: {str(e)}"}


# Utility functions
async def cache_function_result(
    func,
    cache_key: str,
    ttl: int = 3600,
    *args,
    **kwargs
):
    """Cache function result."""
    redis = await get_redis_client()
    cache = RedisCache(redis)
    
    # Check cache first
    cached_result = await cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Execute function and cache result
    result = await func(*args, **kwargs)
    await cache.set(cache_key, result, ttl=ttl)
    
    return result


def cache_decorator(cache_key_func, ttl: int = 3600):
    """Decorator to cache function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_key = cache_key_func(*args, **kwargs)
            return await cache_function_result(
                func,
                cache_key,
                ttl,
                *args,
                **kwargs
            )
        return wrapper
    return decorator 