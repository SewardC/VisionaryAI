"""
Logging utilities for Visionary AI backend services.
"""

import logging
import sys
from typing import Dict, Any
from datetime import datetime

import structlog
from pythonjsonlogger import jsonlogger


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ("json" or "text")
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if log_format == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    if log_format == "json":
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Set up handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls with parameters and execution time."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.utcnow()
        
        logger.info(
            "Function called",
            function=func.__name__,
            args=str(args)[:200],  # Truncate long arguments
            kwargs=str(kwargs)[:200],
            start_time=start_time.isoformat()
        )
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(
                "Function completed",
                function=func.__name__,
                duration_seconds=duration,
                end_time=end_time.isoformat()
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(
                "Function failed",
                function=func.__name__,
                error=str(e),
                error_type=type(e).__name__,
                duration_seconds=duration,
                end_time=end_time.isoformat()
            )
            raise
    
    return wrapper


def log_async_function_call(func):
    """Decorator to log async function calls with parameters and execution time."""
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.utcnow()
        
        logger.info(
            "Async function called",
            function=func.__name__,
            args=str(args)[:200],  # Truncate long arguments
            kwargs=str(kwargs)[:200],
            start_time=start_time.isoformat()
        )
        
        try:
            result = await func(*args, **kwargs)
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(
                "Async function completed",
                function=func.__name__,
                duration_seconds=duration,
                end_time=end_time.isoformat()
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(
                "Async function failed",
                function=func.__name__,
                error=str(e),
                error_type=type(e).__name__,
                duration_seconds=duration,
                end_time=end_time.isoformat()
            )
            raise
    
    return wrapper


class RequestLogger:
    """Context manager for logging HTTP requests."""
    
    def __init__(self, request_id: str, method: str, path: str, user_id: str = None):
        self.request_id = request_id
        self.method = method
        self.path = path
        self.user_id = user_id
        self.logger = get_logger("request")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(
            "Request started",
            request_id=self.request_id,
            method=self.method,
            path=self.path,
            user_id=self.user_id,
            start_time=self.start_time.isoformat()
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(
                "Request completed",
                request_id=self.request_id,
                method=self.method,
                path=self.path,
                user_id=self.user_id,
                duration_seconds=duration,
                end_time=end_time.isoformat()
            )
        else:
            self.logger.error(
                "Request failed",
                request_id=self.request_id,
                method=self.method,
                path=self.path,
                user_id=self.user_id,
                error=str(exc_val),
                error_type=exc_type.__name__,
                duration_seconds=duration,
                end_time=end_time.isoformat()
            )


def log_model_inference(model_name: str, input_size: int, processing_time: float, confidence: float = None):
    """Log model inference metrics."""
    logger = get_logger("model_inference")
    
    log_data = {
        "model_name": model_name,
        "input_size": input_size,
        "processing_time_seconds": processing_time,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if confidence is not None:
        log_data["confidence"] = confidence
    
    logger.info("Model inference completed", **log_data)


def log_security_event(event_type: str, user_id: str = None, ip_address: str = None, details: Dict[str, Any] = None):
    """Log security-related events."""
    logger = get_logger("security")
    
    log_data = {
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if user_id:
        log_data["user_id"] = user_id
    if ip_address:
        log_data["ip_address"] = ip_address
    if details:
        log_data.update(details)
    
    logger.warning("Security event", **log_data)


def log_performance_metric(metric_name: str, value: float, unit: str = None, tags: Dict[str, str] = None):
    """Log performance metrics."""
    logger = get_logger("performance")
    
    log_data = {
        "metric_name": metric_name,
        "value": value,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if unit:
        log_data["unit"] = unit
    if tags:
        log_data["tags"] = tags
    
    logger.info("Performance metric", **log_data) 