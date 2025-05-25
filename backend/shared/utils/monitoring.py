"""
Monitoring and observability utilities for Visionary AI backend services.
"""

import time
from typing import Dict, Any, Optional
from functools import wraps

from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

from shared.config import Settings


# Prometheus metrics
REGISTRY = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

# Model inference metrics
MODEL_INFERENCE_COUNT = Counter(
    'model_inference_total',
    'Total model inferences',
    ['model_name', 'service'],
    registry=REGISTRY
)

MODEL_INFERENCE_DURATION = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_name', 'service'],
    registry=REGISTRY
)

MODEL_INFERENCE_CONFIDENCE = Histogram(
    'model_inference_confidence',
    'Model inference confidence scores',
    ['model_name', 'service'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    registry=REGISTRY
)

# System metrics
ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections',
    ['service'],
    registry=REGISTRY
)

MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['service', 'type'],
    registry=REGISTRY
)

GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id', 'service'],
    registry=REGISTRY
)

GPU_MEMORY_USAGE = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id', 'service'],
    registry=REGISTRY
)

# Processing metrics
DOCUMENTS_PROCESSED = Counter(
    'documents_processed_total',
    'Total documents processed',
    ['service', 'status'],
    registry=REGISTRY
)

PROCESSING_QUEUE_SIZE = Gauge(
    'processing_queue_size',
    'Number of items in processing queue',
    ['service', 'queue_type'],
    registry=REGISTRY
)

# Error metrics
ERROR_COUNT = Counter(
    'errors_total',
    'Total errors',
    ['service', 'error_type'],
    registry=REGISTRY
)


def setup_monitoring(settings: Settings) -> None:
    """
    Set up monitoring and observability.
    
    Args:
        settings: Application settings
    """
    if settings.monitoring.enable_metrics:
        setup_prometheus_metrics(settings)
    
    if settings.monitoring.enable_tracing:
        setup_tracing(settings)
    
    # Instrument FastAPI and other libraries
    FastAPIInstrumentor.instrument()
    RequestsInstrumentor.instrument()
    SQLAlchemyInstrumentor.instrument()


def setup_prometheus_metrics(settings: Settings) -> None:
    """Set up Prometheus metrics collection."""
    # Start Prometheus metrics server
    start_http_server(
        port=settings.monitoring.prometheus_port,
        registry=REGISTRY
    )


def setup_tracing(settings: Settings) -> None:
    """Set up distributed tracing with Jaeger."""
    # Configure tracer provider
    trace.set_tracer_provider(TracerProvider())
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        endpoint=settings.monitoring.jaeger_endpoint,
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name)


def monitor_request(func):
    """Decorator to monitor HTTP requests."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        method = kwargs.get('request', {}).method if 'request' in kwargs else 'UNKNOWN'
        endpoint = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            status_code = getattr(result, 'status_code', 200)
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(time.time() - start_time)
            
            return result
            
        except Exception as e:
            # Record error metrics
            ERROR_COUNT.labels(
                service=endpoint,
                error_type=type(e).__name__
            ).inc()
            
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=500
            ).inc()
            
            raise
    
    return wrapper


def monitor_model_inference(model_name: str, service: str):
    """Decorator to monitor model inference."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics
                MODEL_INFERENCE_COUNT.labels(
                    model_name=model_name,
                    service=service
                ).inc()
                
                MODEL_INFERENCE_DURATION.labels(
                    model_name=model_name,
                    service=service
                ).observe(duration)
                
                # Record confidence if available
                if hasattr(result, 'confidence') and result.confidence is not None:
                    MODEL_INFERENCE_CONFIDENCE.labels(
                        model_name=model_name,
                        service=service
                    ).observe(result.confidence)
                
                return result
                
            except Exception as e:
                ERROR_COUNT.labels(
                    service=service,
                    error_type=type(e).__name__
                ).inc()
                raise
        
        return wrapper
    return decorator


def record_document_processed(service: str, status: str) -> None:
    """Record document processing metrics."""
    DOCUMENTS_PROCESSED.labels(
        service=service,
        status=status
    ).inc()


def update_queue_size(service: str, queue_type: str, size: int) -> None:
    """Update processing queue size metrics."""
    PROCESSING_QUEUE_SIZE.labels(
        service=service,
        queue_type=queue_type
    ).set(size)


def update_active_connections(service: str, count: int) -> None:
    """Update active connections metrics."""
    ACTIVE_CONNECTIONS.labels(service=service).set(count)


def update_memory_usage(service: str, memory_type: str, bytes_used: int) -> None:
    """Update memory usage metrics."""
    MEMORY_USAGE.labels(
        service=service,
        type=memory_type
    ).set(bytes_used)


def update_gpu_metrics(gpu_id: str, service: str, utilization: float, memory_used: int) -> None:
    """Update GPU utilization and memory metrics."""
    GPU_UTILIZATION.labels(
        gpu_id=gpu_id,
        service=service
    ).set(utilization)
    
    GPU_MEMORY_USAGE.labels(
        gpu_id=gpu_id,
        service=service
    ).set(memory_used)


class MetricsCollector:
    """Collect and report system metrics."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
    
    def collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        import psutil
        
        # Memory usage
        memory = psutil.virtual_memory()
        update_memory_usage(
            self.service_name,
            "system",
            memory.used
        )
        
        # CPU usage would be collected here
        # GPU metrics would be collected here if available
    
    def collect_gpu_metrics(self) -> None:
        """Collect GPU metrics if available."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # Get GPU utilization (would need nvidia-ml-py for real metrics)
                    memory_used = torch.cuda.memory_allocated(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    utilization = (memory_used / memory_total) * 100
                    
                    update_gpu_metrics(
                        gpu_id=str(i),
                        service=self.service_name,
                        utilization=utilization,
                        memory_used=memory_used
                    )
        except ImportError:
            pass


class TracingContext:
    """Context manager for distributed tracing."""
    
    def __init__(self, operation_name: str, service_name: str, **attributes):
        self.operation_name = operation_name
        self.service_name = service_name
        self.attributes = attributes
        self.tracer = get_tracer(service_name)
        self.span = None
    
    def __enter__(self):
        self.span = self.tracer.start_span(self.operation_name)
        
        # Add attributes
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.span.set_attribute("error", True)
            self.span.set_attribute("error.type", exc_type.__name__)
            self.span.set_attribute("error.message", str(exc_val))
        
        self.span.end()


def trace_function(operation_name: str = None, service_name: str = None):
    """Decorator to trace function execution."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            svc_name = service_name or func.__module__
            
            with TracingContext(op_name, svc_name) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    raise
        
        return wrapper
    return decorator


class HealthChecker:
    """Health check utilities."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.checks = {}
    
    def add_check(self, name: str, check_func):
        """Add a health check function."""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                results[name] = {"status": "healthy", "details": result}
            except Exception as e:
                results[name] = {"status": "unhealthy", "error": str(e)}
        
        return results 