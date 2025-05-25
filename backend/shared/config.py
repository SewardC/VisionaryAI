"""
Centralized configuration management for Visionary AI backend services.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql://visionary:dev_password@localhost:5432/visionary_dev",
        env="DATABASE_URL"
    )
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    echo: bool = Field(default=False, env="DB_ECHO")


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    decode_responses: bool = Field(default=True, env="REDIS_DECODE_RESPONSES")


class QdrantSettings(BaseSettings):
    """Qdrant vector database settings."""
    
    url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    timeout: int = Field(default=30, env="QDRANT_TIMEOUT")


class AWSSettings(BaseSettings):
    """AWS configuration settings."""
    
    region: str = Field(default="us-west-2", env="AWS_REGION")
    access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    s3_bucket: str = Field(default="visionary-ai-documents", env="AWS_S3_BUCKET")
    
    # DynamoDB tables
    documents_table: str = Field(default="visionary-documents", env="DYNAMODB_DOCUMENTS_TABLE")
    users_table: str = Field(default="visionary-users", env="DYNAMODB_USERS_TABLE")
    
    # Neptune
    neptune_endpoint: Optional[str] = Field(default=None, env="NEPTUNE_ENDPOINT")


class ModelSettings(BaseSettings):
    """AI model configuration settings."""
    
    # OCR Model
    ocr_model_name: str = Field(default="allenai/olmocr-7b", env="OCR_MODEL_NAME")
    ocr_model_path: Optional[str] = Field(default=None, env="OCR_MODEL_PATH")
    ocr_device: str = Field(default="cuda", env="OCR_DEVICE")
    ocr_batch_size: int = Field(default=4, env="OCR_BATCH_SIZE")
    
    # LLM Model
    llm_model_name: str = Field(default="Qwen/Qwen2-VL-7B-Instruct", env="LLM_MODEL_NAME")
    llm_model_path: Optional[str] = Field(default=None, env="LLM_MODEL_PATH")
    llm_device: str = Field(default="cuda", env="LLM_DEVICE")
    llm_max_tokens: int = Field(default=2048, env="LLM_MAX_TOKENS")
    
    # Embedding Model
    embedding_model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL_NAME"
    )
    
    # Model optimization
    use_quantization: bool = Field(default=True, env="USE_QUANTIZATION")
    quantization_bits: int = Field(default=8, env="QUANTIZATION_BITS")
    use_flash_attention: bool = Field(default=True, env="USE_FLASH_ATTENTION")


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""
    
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        env="CORS_ORIGINS"
    )
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    
    # Prometheus
    prometheus_port: int = Field(default=8080, env="PROMETHEUS_PORT")
    
    # Jaeger
    jaeger_endpoint: str = Field(
        default="http://localhost:14268/api/traces",
        env="JAEGER_ENDPOINT"
    )
    
    # Log level
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")


class PerformanceSettings(BaseSettings):
    """Performance optimization settings."""
    
    # API rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # Processing limits
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    max_pages_per_document: int = Field(default=100, env="MAX_PAGES_PER_DOCUMENT")
    
    # Caching
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    
    # GPU settings
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    enable_gpu_optimization: bool = Field(default=True, env="ENABLE_GPU_OPTIMIZATION")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application info
    app_name: str = Field(default="Visionary AI", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Service settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    qdrant: QdrantSettings = QdrantSettings()
    aws: AWSSettings = AWSSettings()
    models: ModelSettings = ModelSettings()
    security: SecuritySettings = SecuritySettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    performance: PerformanceSettings = PerformanceSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Service-specific settings for different microservices
class OCRServiceSettings(Settings):
    """OCR service specific settings."""
    
    service_name: str = "ocr-service"
    port: int = Field(default=8001, env="OCR_SERVICE_PORT")


class LLMServiceSettings(Settings):
    """LLM service specific settings."""
    
    service_name: str = "llm-service"
    port: int = Field(default=8002, env="LLM_SERVICE_PORT")


class GatewaySettings(Settings):
    """API Gateway specific settings."""
    
    service_name: str = "api-gateway"
    port: int = Field(default=8000, env="GATEWAY_PORT")
    
    # Service URLs
    ocr_service_url: str = Field(
        default="http://localhost:8001",
        env="OCR_SERVICE_URL"
    )
    llm_service_url: str = Field(
        default="http://localhost:8002",
        env="LLM_SERVICE_URL"
    )


# Factory functions for service-specific settings
@lru_cache()
def get_ocr_settings() -> OCRServiceSettings:
    """Get OCR service settings."""
    return OCRServiceSettings()


@lru_cache()
def get_llm_settings() -> LLMServiceSettings:
    """Get LLM service settings."""
    return LLMServiceSettings()


@lru_cache()
def get_gateway_settings() -> GatewaySettings:
    """Get API Gateway settings."""
    return GatewaySettings() 