"""
OCR Service - Visionary AI
Handles document text extraction using AllenAI olmOCR model.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from uuid import UUID

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_bytes

from shared.config import get_ocr_settings, OCRServiceSettings
from shared.models import (
    OCRResult, DocumentResponse, HealthResponse, ErrorResponse,
    Language, DocumentStatus
)
from shared.utils.logging import setup_logging
from shared.utils.monitoring import setup_monitoring
from shared.utils.database import get_database
from shared.utils.redis import get_redis_client


# Global variables for model and processor
model = None
processor = None
device = None


class OCRProcessor:
    """OCR processing class using olmOCR model."""
    
    def __init__(self, settings: OCRServiceSettings):
        self.settings = settings
        self.model = None
        self.processor = None
        self.device = None
        
    async def initialize(self):
        """Initialize the OCR model and processor."""
        global model, processor, device
        
        try:
            # Set device
            if torch.cuda.is_available() and self.settings.models.ocr_device == "cuda":
                device = torch.device("cuda")
                logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logging.info("Using CPU for OCR processing")
            
            # Load model and processor
            model_name = self.settings.models.ocr_model_name
            logging.info(f"Loading OCR model: {model_name}")
            
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto" if device.type == "cuda" else None,
                trust_remote_code=True
            )
            
            # Apply quantization if enabled
            if self.settings.models.use_quantization and device.type == "cuda":
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=self.settings.models.quantization_bits == 8,
                    load_in_4bit=self.settings.models.quantization_bits == 4,
                )
                model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            model.eval()
            logging.info("OCR model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize OCR model: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Apply image enhancement techniques
        # 1. Noise reduction
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # 2. Contrast enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img_array = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(img_array, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL Image
        return Image.fromarray(img_array)
    
    def detect_language(self, text: str) -> Optional[Language]:
        """Detect language from extracted text."""
        # Simple heuristic-based language detection
        # In production, you might want to use a proper language detection library
        
        if not text.strip():
            return None
        
        # Count characters from different scripts
        latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 256)
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
        
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return None
        
        # Determine dominant script
        if chinese_chars / total_chars > 0.3:
            return Language.CHINESE
        elif korean_chars / total_chars > 0.3:
            return Language.KOREAN
        elif japanese_chars / total_chars > 0.3:
            return Language.JAPANESE
        elif latin_chars / total_chars > 0.7:
            # Could be English or Spanish, default to English
            return Language.ENGLISH
        
        return Language.ENGLISH  # Default fallback
    
    async def process_image(self, image: Image.Image, page_number: int = 1) -> OCRResult:
        """Process a single image and extract text."""
        try:
            start_time = time.time()
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Prepare inputs
            inputs = processor(processed_image, return_tensors="pt").to(device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=0.0,
                )
            
            # Decode the generated text
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            # Extract text and confidence (simplified)
            # In a real implementation, you'd extract bounding boxes and confidence scores
            text = generated_text.strip()
            confidence = 0.95  # Placeholder - would come from model output
            
            # Detect language
            detected_language = self.detect_language(text)
            
            processing_time = time.time() - start_time
            logging.info(f"OCR processed page {page_number} in {processing_time:.2f}s")
            
            return OCRResult(
                page_number=page_number,
                text=text,
                confidence=confidence,
                bounding_boxes=[],  # Would be populated with actual bounding boxes
                layout_info={"processing_time": processing_time},
                language_detected=detected_language
            )
            
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    
    async def process_pdf(self, pdf_bytes: bytes) -> List[OCRResult]:
        """Process a PDF document and extract text from all pages."""
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes, dpi=300)
            
            results = []
            for i, image in enumerate(images, 1):
                result = await self.process_image(image, page_number=i)
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing PDF: {e}")
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


# Initialize OCR processor
ocr_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global ocr_processor
    
    # Startup
    settings = get_ocr_settings()
    setup_logging(settings.monitoring.log_level)
    setup_monitoring(settings)
    
    ocr_processor = OCRProcessor(settings)
    await ocr_processor.initialize()
    
    logging.info("OCR Service started successfully")
    yield
    
    # Shutdown
    logging.info("OCR Service shutting down")


# Create FastAPI app
app = FastAPI(
    title="Visionary AI - OCR Service",
    description="High-accuracy multilingual OCR service using AllenAI olmOCR",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
settings = get_ocr_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get OCR processor
def get_ocr_processor() -> OCRProcessor:
    """Get OCR processor instance."""
    if ocr_processor is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")
    return ocr_processor


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    dependencies = {
        "model": "loaded" if model is not None else "not_loaded",
        "device": str(device) if device is not None else "unknown",
        "cuda_available": str(torch.cuda.is_available())
    }
    
    return HealthResponse(
        service="ocr-service",
        dependencies=dependencies
    )


# OCR endpoints
@app.post("/ocr/image", response_model=OCRResult)
async def process_image_endpoint(
    file: UploadFile = File(...),
    page_number: int = 1,
    processor: OCRProcessor = Depends(get_ocr_processor)
):
    """Process a single image file and extract text."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        result = await processor.process_image(image, page_number)
        return result
        
    except Exception as e:
        logging.error(f"Error in process_image_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/pdf", response_model=List[OCRResult])
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    processor: OCRProcessor = Depends(get_ocr_processor)
):
    """Process a PDF file and extract text from all pages."""
    try:
        # Validate file type
        if file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read and process PDF
        pdf_bytes = await file.read()
        results = await processor.process_pdf(pdf_bytes)
        
        return results
        
    except Exception as e:
        logging.error(f"Error in process_pdf_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/document/{document_id}")
async def process_document_endpoint(
    document_id: UUID,
    processor: OCRProcessor = Depends(get_ocr_processor),
    db = Depends(get_database)
):
    """Process a document by ID (fetch from S3 and process)."""
    try:
        # This would fetch the document from S3 and process it
        # Implementation depends on your S3 integration
        pass
        
    except Exception as e:
        logging.error(f"Error in process_document_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch processing endpoint
@app.post("/ocr/batch", response_model=List[OCRResult])
async def process_batch_endpoint(
    files: List[UploadFile] = File(...),
    processor: OCRProcessor = Depends(get_ocr_processor)
):
    """Process multiple files in batch."""
    try:
        results = []
        
        for i, file in enumerate(files):
            if file.content_type.startswith('image/'):
                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes))
                result = await processor.process_image(image, page_number=i+1)
                results.append(result)
            elif file.content_type == 'application/pdf':
                pdf_bytes = await file.read()
                pdf_results = await processor.process_pdf(pdf_bytes)
                results.extend(pdf_results)
        
        return results
        
    except Exception as e:
        logging.error(f"Error in process_batch_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import io
    
    settings = get_ocr_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.monitoring.log_level.lower()
    ) 