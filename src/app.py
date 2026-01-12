import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import pickle
import os

# Load the trained model and tokenizer
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Load model and tokenizer
    model = load_model('models/imdb_arabic.keras')
    
    # Load tokenizer (assuming it was saved as .pkl)
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    print("✅ Model and tokenizer loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    tokenizer = None

# Configuration
MAX_LEN = 500
MAX_FEATURES = 5000

# Define request/response schemas
class ReviewRequest(BaseModel):
    review: str
    language: str = "arabic"  # arabic, english, or auto-detect

class SentimentResponse(BaseModel):
    sentiment: str  # positive or negative
    confidence: float
    sentiment_score: float
    processing_time_ms: float

# Arabic preprocessing function (simplified)
def preprocess_arabic_text(text: str) -> str:
    """Basic Arabic text preprocessing"""
    if not text or not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = text.strip()
    
    # Remove diacritics and special characters (simplified)
    import re
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]', '', text)
    
    return text

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

def predict_sentiment(text: str) -> Dict[str, Any]:
    """Predict sentiment for a given text"""
    if model is None or tokenizer is None:
        raise ValueError("Model not loaded properly")
    
    if not text or len(text.strip()) == 0:
        return {"sentiment": "neutral", "confidence": 0.5, "sentiment_score": 0.0}
    
    # Preprocess Arabic text
    processed_text = preprocess_arabic_text(text)
    
    if not processed_text:
        return {"sentiment": "neutral", "confidence": 0.5, "sentiment_score": 0.0}
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0][0]
    
    # Determine sentiment
    sentiment = "positive" if prediction >= 0.5 else "negative"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    
    return {
        "sentiment": sentiment,
        "confidence": float(confidence),
        "sentiment_score": float(prediction),
        "original_text": text,
        "processed_text": processed_text
    }

# Health check endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Arabic Sentiment Analyzer API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model": "loaded",
        "max_sequence_length": MAX_LEN,
        "max_features": MAX_FEATURES
    }

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "imdb_arabic",
        "architecture": "Embedding-LSTM-Dense",
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "layers": len(model.layers)
    }

@app.post("/predict", response_model=SentimentResponse, tags=["Prediction"])
async def predict(request: ReviewRequest):
    """Predict sentiment for a movie review"""
    import time
    
    start_time = time.time()
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    try:
        # Predict sentiment
        result = predict_sentiment(request.review)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        return SentimentResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            sentiment_score=result["sentiment_score"],
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
