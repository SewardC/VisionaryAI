"""
Shared data models and schemas for Visionary AI backend services.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, DateTime, Text, JSON, Float, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func


# SQLAlchemy Base
Base = declarative_base()


# Enums
class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentType(str, Enum):
    """Document type classification."""
    PDF = "pdf"
    IMAGE = "image"
    SCAN = "scan"
    UNKNOWN = "unknown"


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    CHINESE = "zh"
    KOREAN = "ko"
    JAPANESE = "ja"
    SPANISH = "es"


class EntityType(str, Enum):
    """Entity types for relationship graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    DOCUMENT = "document"
    OTHER = "other"


class ProcessingStage(str, Enum):
    """Processing pipeline stages."""
    UPLOAD = "upload"
    OCR = "ocr"
    EMBEDDING = "embedding"
    ENTITY_EXTRACTION = "entity_extraction"
    GRAPH_UPDATE = "graph_update"
    INDEXING = "indexing"
    COMPLETED = "completed"


# Pydantic Models for API
class BaseSchema(BaseModel):
    """Base schema with common fields."""
    
    class Config:
        orm_mode = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class DocumentMetadata(BaseSchema):
    """Document metadata schema."""
    
    filename: str
    file_size: int
    mime_type: str
    page_count: Optional[int] = None
    language: Optional[Language] = None
    upload_timestamp: datetime
    processing_time: Optional[float] = None


class OCRResult(BaseSchema):
    """OCR processing result."""
    
    page_number: int
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_boxes: List[Dict[str, Any]] = Field(default_factory=list)
    layout_info: Optional[Dict[str, Any]] = None
    language_detected: Optional[Language] = None


class EmbeddingResult(BaseSchema):
    """Text embedding result."""
    
    chunk_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityResult(BaseSchema):
    """Named entity extraction result."""
    
    text: str
    label: EntityType
    confidence: float = Field(ge=0.0, le=1.0)
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelationshipResult(BaseSchema):
    """Entity relationship result."""
    
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    context: Optional[str] = None


class DocumentCreate(BaseSchema):
    """Document creation schema."""
    
    filename: str
    file_size: int
    mime_type: str
    s3_key: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentUpdate(BaseSchema):
    """Document update schema."""
    
    status: Optional[DocumentStatus] = None
    metadata: Optional[Dict[str, Any]] = None
    ocr_results: Optional[List[OCRResult]] = None
    entities: Optional[List[EntityResult]] = None
    processing_error: Optional[str] = None


class DocumentResponse(BaseSchema):
    """Document response schema."""
    
    id: UUID
    filename: str
    status: DocumentStatus
    document_type: DocumentType
    language: Optional[Language]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    ocr_results: List[OCRResult] = Field(default_factory=list)
    entities: List[EntityResult] = Field(default_factory=list)
    processing_stages: Dict[ProcessingStage, bool] = Field(default_factory=dict)


class QueryRequest(BaseSchema):
    """Query request schema."""
    
    query: str
    document_ids: Optional[List[UUID]] = None
    language: Optional[Language] = None
    max_results: int = Field(default=10, ge=1, le=100)
    include_context: bool = Field(default=True)


class QueryResponse(BaseSchema):
    """Query response schema."""
    
    query: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseSchema):
    """Graph node schema."""
    
    id: str
    label: str
    type: EntityType
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseSchema):
    """Graph edge schema."""
    
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphResponse(BaseSchema):
    """Graph query response schema."""
    
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# SQLAlchemy Models
class Document(Base):
    """Document database model."""
    
    __tablename__ = "documents"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    filename = Column(String(255), nullable=False)
    s3_key = Column(String(500), nullable=False, unique=True)
    status = Column(String(50), nullable=False, default=DocumentStatus.UPLOADED)
    document_type = Column(String(50), nullable=False, default=DocumentType.UNKNOWN)
    language = Column(String(10), nullable=True)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    page_count = Column(Integer, nullable=True)
    
    # Processing information
    processing_stages = Column(JSON, nullable=False, default=dict)
    processing_error = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)
    
    # OCR results
    ocr_results = Column(JSON, nullable=False, default=list)
    
    # Entity extraction results
    entities = Column(JSON, nullable=False, default=list)
    
    # Metadata
    metadata = Column(JSON, nullable=False, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ProcessingJob(Base):
    """Processing job tracking model."""
    
    __tablename__ = "processing_jobs"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(PGUUID(as_uuid=True), nullable=False)
    stage = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=False, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class User(Base):
    """User model for authentication."""
    
    __tablename__ = "users"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    # Metadata
    metadata = Column(JSON, nullable=False, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# API Response Models
class HealthResponse(BaseSchema):
    """Health check response."""
    
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "0.1.0"
    service: str
    dependencies: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseSchema):
    """Error response schema."""
    
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SuccessResponse(BaseSchema):
    """Generic success response."""
    
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Validation helpers
def validate_file_size(file_size: int, max_size_mb: int = 50) -> bool:
    """Validate file size."""
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes


def validate_mime_type(mime_type: str) -> bool:
    """Validate supported MIME types."""
    supported_types = [
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/bmp",
        "image/webp"
    ]
    return mime_type in supported_types


# Custom validators
class DocumentCreateValidator(DocumentCreate):
    """Document creation with validation."""
    
    @validator("file_size")
    def validate_file_size(cls, v):
        if not validate_file_size(v):
            raise ValueError("File size exceeds maximum allowed size")
        return v
    
    @validator("mime_type")
    def validate_mime_type(cls, v):
        if not validate_mime_type(v):
            raise ValueError("Unsupported file type")
        return v 