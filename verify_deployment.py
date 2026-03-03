#!/usr/bin/env python3
"""
Deployment Verification Script
This script verifies that the Emotion API is functioning exactly as documented in the README.
It tests the API endpoints without loading the heavy LLM (using the test_main.py mock server),
sends a POST request, and validates that the stripped-down JSON response (emotion only)
is correctly returned without causing Pydantic validation errors.
"""

import subprocess
import time
import urllib.request
import urllib.error
import json
import sys
import os

API_URL = "http://127.0.0.1:8000/predict"

def verify_api():
    print("🚀 开始环境部署验证测试 (Ver 2.0)")
    
    # 1. Start the mock FastAPI server
    os.chdir("Emotion_Ai_Api_CPU")
    print("▶️ 正在启动 FastAPI 测试桩...")
    
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.test_main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give the server a few seconds to boot up
    time.sleep(3)
    
    if server_process.poll() is not None:
        print("❌ 错误：FastAPI 服务器启动失败！")
        print(server_process.stderr.read().decode('utf-8', errors='ignore'))
        sys.exit(1)
        
    print("✅ FastAPI 服务器启动成功")

    # 2. Send the POST request as documented in the README
    payload = {
        "text": "我都走到半路了，突然下大暴雨，全身都淋湿了！",
        "history": [
            "今天天气预报明明说没雨的",
            "是啊，所以我没带伞"
        ]
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(API_URL, data=data, headers={'Content-Type': 'application/json'})
    
    print("▶️ 正在发送符合 README 规范的推理请求...")
    try:
        with urllib.request.urlopen(req) as response:
            status_code = response.getcode()
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            
            print(f"✅ 收到响应 (HTTP {status_code}): {res_body}")
            
            # 3. Validate the payload schema (should ONLY have 'emotion', NO 'impact' or 'speaker')
            if "emotion" not in res_json:
                print("❌ 验证异常：返回体中缺失 'emotion' 字段！")
                server_process.kill()
                sys.exit(1)
                
            if "speaker" in res_json or "impact" in res_json:
                print("❌ 验证异常：返回体中仍然包含被废弃的 'speaker' 或 'impact' 字段！")
                server_process.kill()
                sys.exit(1)
                
            print("🎉 验证通过：API 返回体字段严格遵循最新的精简规范（仅含 emotion），所有瘦身修改均已生效且未报错！")

    except urllib.error.URLError as e:
        print(f"❌ 请求失败: {e}")
        server_process.kill()
        sys.exit(1)
        
    # Cleanup
    server_process.kill()
    print("🏁 测试结束，已安全停止服务器。验证脚本跑通！")

if __name__ == "__main__":
    verify_api()
