import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import pickle
import sys
import os

# IMPORT PREDICT FILE :
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from models.predict import predict_review




# Define request/response schemas
class ReviewRequest(BaseModel):
    review: str
    language: str = "arabic"  # arabic, english, or auto-detect

class SentimentResponse(BaseModel):
    sentiment: str  # positive or negative
    confidence: float
   

# Initialize FastAPI app
app = FastAPI(
    title="Arabic Sentiment Analyzer API",
    description="Sentiment analysis for Moroccan Arabic film reviews",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Arabic Sentiment Analyzer API",
        "status": "running"
    }


@app.post("/predict", response_model=SentimentResponse, tags=["Prediction"])
async def predict(request: ReviewRequest):
    """Predict sentiment for a movie review"""
    import time
    
    start_time = time.time()
   
    
    try:
        # Predict sentiment
      
        result = predict_review(request.review)    
        sentiment, confidence = result 
        processing_time_ms = (time.time() - start_time) * 1000

        return SentimentResponse(
            sentiment=sentiment,  # First element of tuple
            confidence=confidence,  # Second element of tuple
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
