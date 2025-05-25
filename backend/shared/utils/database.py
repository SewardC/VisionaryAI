"""
Database utilities for Visionary AI backend services.
"""

import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from shared.config import get_settings
from shared.models import Base


# Global database engine and session maker
engine = None
async_session_maker = None


async def init_database():
    """Initialize database connection and create tables."""
    global engine, async_session_maker
    
    settings = get_settings()
    
    # Create async engine
    engine = create_async_engine(
        settings.database.url.replace("postgresql://", "postgresql+asyncpg://"),
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        echo=settings.database.echo,
        future=True
    )
    
    # Create session maker
    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_database():
    """Close database connections."""
    global engine
    if engine:
        await engine.dispose()


@asynccontextmanager
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session context manager."""
    if async_session_maker is None:
        await init_database()
    
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_database() -> AsyncSession:
    """Dependency to get database session."""
    async with get_database_session() as session:
        yield session


class DatabaseManager:
    """Database manager for handling connections and operations."""
    
    def __init__(self):
        self.engine = None
        self.session_maker = None
    
    async def initialize(self, database_url: str, **kwargs):
        """Initialize database connection."""
        self.engine = create_async_engine(
            database_url.replace("postgresql://", "postgresql+asyncpg://"),
            **kwargs
        )
        
        self.session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session."""
        async with self.session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


class BaseRepository:
    """Base repository class with common database operations."""
    
    def __init__(self, session: AsyncSession, model_class):
        self.session = session
        self.model_class = model_class
    
    async def create(self, **kwargs):
        """Create a new record."""
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance
    
    async def get_by_id(self, id):
        """Get record by ID."""
        return await self.session.get(self.model_class, id)
    
    async def get_all(self, limit: int = 100, offset: int = 0):
        """Get all records with pagination."""
        from sqlalchemy import select
        
        stmt = select(self.model_class).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def update(self, id, **kwargs):
        """Update record by ID."""
        instance = await self.get_by_id(id)
        if instance:
            for key, value in kwargs.items():
                setattr(instance, key, value)
            await self.session.flush()
            await self.session.refresh(instance)
        return instance
    
    async def delete(self, id):
        """Delete record by ID."""
        instance = await self.get_by_id(id)
        if instance:
            await self.session.delete(instance)
            await self.session.flush()
        return instance
    
    async def exists(self, **filters):
        """Check if record exists with given filters."""
        from sqlalchemy import select, exists
        
        stmt = select(exists().where(
            *[getattr(self.model_class, key) == value for key, value in filters.items()]
        ))
        result = await self.session.execute(stmt)
        return result.scalar()
    
    async def count(self, **filters):
        """Count records with given filters."""
        from sqlalchemy import select, func
        
        stmt = select(func.count(self.model_class.id))
        if filters:
            stmt = stmt.where(
                *[getattr(self.model_class, key) == value for key, value in filters.items()]
            )
        
        result = await self.session.execute(stmt)
        return result.scalar()


class DocumentRepository(BaseRepository):
    """Repository for document operations."""
    
    def __init__(self, session: AsyncSession):
        from shared.models import Document
        super().__init__(session, Document)
    
    async def get_by_s3_key(self, s3_key: str):
        """Get document by S3 key."""
        from sqlalchemy import select
        
        stmt = select(self.model_class).where(self.model_class.s3_key == s3_key)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by_status(self, status: str, limit: int = 100):
        """Get documents by status."""
        from sqlalchemy import select
        
        stmt = select(self.model_class).where(
            self.model_class.status == status
        ).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def update_processing_stage(self, document_id, stage: str, completed: bool = True):
        """Update processing stage for document."""
        document = await self.get_by_id(document_id)
        if document:
            if not document.processing_stages:
                document.processing_stages = {}
            document.processing_stages[stage] = completed
            await self.session.flush()
            await self.session.refresh(document)
        return document


class UserRepository(BaseRepository):
    """Repository for user operations."""
    
    def __init__(self, session: AsyncSession):
        from shared.models import User
        super().__init__(session, User)
    
    async def get_by_email(self, email: str):
        """Get user by email."""
        from sqlalchemy import select
        
        stmt = select(self.model_class).where(self.model_class.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_active_users(self):
        """Get all active users."""
        from sqlalchemy import select
        
        stmt = select(self.model_class).where(self.model_class.is_active == True)
        result = await self.session.execute(stmt)
        return result.scalars().all()


class ProcessingJobRepository(BaseRepository):
    """Repository for processing job operations."""
    
    def __init__(self, session: AsyncSession):
        from shared.models import ProcessingJob
        super().__init__(session, ProcessingJob)
    
    async def get_by_document_id(self, document_id):
        """Get processing jobs by document ID."""
        from sqlalchemy import select
        
        stmt = select(self.model_class).where(
            self.model_class.document_id == document_id
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_pending_jobs(self, stage: str = None):
        """Get pending processing jobs."""
        from sqlalchemy import select
        
        stmt = select(self.model_class).where(self.model_class.status == "pending")
        if stage:
            stmt = stmt.where(self.model_class.stage == stage)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def update_job_status(self, job_id, status: str, error_message: str = None):
        """Update job status."""
        from datetime import datetime
        
        job = await self.get_by_id(job_id)
        if job:
            job.status = status
            if status == "running" and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in ["completed", "failed"]:
                job.completed_at = datetime.utcnow()
            
            if error_message:
                job.error_message = error_message
            
            await self.session.flush()
            await self.session.refresh(job)
        
        return job


# Health check function for database
async def check_database_health() -> dict:
    """Check database connectivity."""
    try:
        async with get_database_session() as session:
            # Simple query to check connectivity
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            result.scalar()
            return {"status": "healthy", "message": "Database connection successful"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Database connection failed: {str(e)}"}


# Migration utilities
async def run_migrations():
    """Run database migrations."""
    # This would integrate with Alembic for production
    # For now, just create tables
    await init_database()


async def reset_database():
    """Reset database (development only)."""
    global engine
    
    if engine:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all) 