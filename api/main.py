from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import YouTubeSentimentPipeline

DEFAULT_API_KEY = "AIzaSyCwzrJzwNzfje1xJZmZmBU2L2O-cHWPHk0"
API_KEY = "YOUR_YOUTUBE_API_KEY"

FINAL_API_KEY = API_KEY if API_KEY != "YOUR_YOUTUBE_API_KEY" else DEFAULT_API_KEY
MODEL_PATH = "roberta-goemotions-model-new"

app = FastAPI(title="YouTube Emotion API")
pipeline = YouTubeSentimentPipeline(model_path=MODEL_PATH, developer_key=FINAL_API_KEY)

class VideoRequest(BaseModel):
    url: str

@app.get("/")
def home():
    return {"status": "Active", "endpoints": "/analyze_video"}

@app.post("/analyze_video")
def analyze_video(payload: VideoRequest):
    result = pipeline.analyze_youtube_video(
        url=payload.url, 
        max_comments=20, 
        threshold=0.05
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result