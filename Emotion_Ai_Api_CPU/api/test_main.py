from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import random
import asyncio

app = FastAPI(title="情绪识别 UI 本地测试", description="这是一个不加载模型的伪测试桩", version="1.0")

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


# ── 测试用的模拟数据列表 ──
MOCK_EMOTIONS = [
    "joy", "sadness", "anger", "fear", "surprise", 
    "disgust", "anticipation", "trust", "love", "amusement",
    "optimism", "annoyance", "curiosity", "disappointment"
]


@app.post("/predict", response_model=EmotionResponse)
async def predict(request: EmotionRequest):
    """
    不调用任何真实模型，而是模拟一个处理延迟并返回随机情绪的接口
    """
    # 模拟真实模型的计算耗时 (假设 1 到 3 秒)
    delay = random.uniform(1.0, 3.0)
    print(f"[TEST 模式] 收到请求: {request.text}")
    print(f"[TEST 模式] 对话历史: {request.history}")
    print(f"[TEST 模式] 模拟处理耗时: {delay:.2f} 秒...")
    
    await asyncio.sleep(delay)
    
    # 简单随机返回几个结果
    emotion = random.choice(MOCK_EMOTIONS)
    
    # 彩蛋：如果是点击内置 example 发过来的特定文本，返回固定的符合预期的测试数据
    lower_text = request.text.lower()
    if "promoted" in lower_text:
        emotion = "joy"
    elif "wallet" in lower_text:
        emotion = "sadness"
    elif "spider" in lower_text:
        emotion = "fear"
    elif "scared me" in lower_text:
        emotion = "fear"
        
    return EmotionResponse(
        emotion=emotion
    )


# ── 前端页面 ──
STATIC_DIR = Path(__file__).parent / "static"
# 确保在运行测试脚本前静态目录已经存在
STATIC_DIR.mkdir(exist_ok=True)

@app.get("/")
async def index():
    """返回交互式前端页面"""
    return FileResponse(STATIC_DIR / "index.html")

# 挂载真实的静态文件前端
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    # 直接运行即可：python api/test_main.py 
    print("="*60)
    print("🚀 启动纯前端测试模式 (不加载 AI 模型)")
    print("🌐 请在浏览器打开: http://127.0.0.1:8000")
    print("="*60)
    uvicorn.run(app, host="127.0.0.1", port=8000)
