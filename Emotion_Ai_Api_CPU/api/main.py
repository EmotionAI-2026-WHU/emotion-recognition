from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from . import model_inference

app = FastAPI(title="细粒度情绪识别 API", description="基于 Qwen2.5-7B 的对话情绪识别", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmotionRequest(BaseModel):
    text: str
    history: Optional[List[str]] = []

class EmotionResponse(BaseModel):
    emotion: str

@app.on_event("startup")
async def startup_event():
    model_inference.load_model()

# ── API 端点 ──
@app.post("/predict", response_model=EmotionResponse)
async def predict(request: EmotionRequest):
    result = model_inference.predict(request.text, request.history)
    return result

# ── 前端页面 ──
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

@app.get("/")
async def index():
    """返回交互式前端页面"""
    return FileResponse(STATIC_DIR / "index.html")

# 挂载静态文件目录 (CSS/JS/图片等)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")