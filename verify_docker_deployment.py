#!/usr/bin/env python3
"""
Docker Deployment Verification Script
This script verifies that the API can be fully containerized and booted via Docker Compose
WITHOUT triggering the actual heavy LLM downloads that would normally occur.

It dynamically injects a `docker-compose.override.yml` to redirect the entrypoint to the
mock server (`test_main.py`), boots the containers, verified connection, and then tears down.
"""

import subprocess
import time
import urllib.request
import urllib.error
import json
import sys
import os

API_URL = "http://127.0.0.1:8000/predict"
OVERRIDE_FILE = "Emotion_Ai_Api_CPU/docker-compose.override.yml"

OVERRIDE_CONTENT = """
version: '3.8'
services:
  web:
    # 覆盖原脚本的 command，转而启动体积极小、无需下载大模型的测试桩
    command: uvicorn api.test_main:app --host 0.0.0.0 --port 8000
    # 由于是在宿主机做端口探活，为了不冲突，我们可以映射到 8000
    ports:
      - "8000:8000"
"""

def main():
    print("🐳 开始 Docker 环境部署验证测试 (Mock 模式)")

    # 1. Create the override file to avoid loading heavy models
    with open(OVERRIDE_FILE, "w", encoding="utf-8") as f:
        f.write(OVERRIDE_CONTENT)
    print(f"✅ 生成配置覆盖文件 (指向 test_main.py): {OVERRIDE_FILE}")

    # 2. Start the Docker containers
    os.chdir("Emotion_Ai_Api_CPU")
    print("▶️ 正在构建并启动 Docker 容器 (这可能需要花费数十秒构建镜像)...")
    
    # We use docker-compose up -d --build to run in detached mode
    up_process = subprocess.run(
        "docker-compose up -d --build",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    
    if up_process.returncode != 0:
        print("❌ 错误：Docker 容器启动失败！")
        print(up_process.stderr)
        os.remove("docker-compose.override.yml")
        sys.exit(1)
        
    print("✅ Docker 容器构建并启动成功")

    # Give the containerized uvicorn a moment to boot
    time.sleep(5)

    # 3. Send the POST request to the Docker-hosted API
    payload = {
        "text": "Docker 里的网络连通性测试！",
        "history": ["你好容器"]
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(API_URL, data=data, headers={'Content-Type': 'application/json'})
    
    print("▶️ 正在向 Docker 容器内的 API 发送推理请求...")
    try:
        with urllib.request.urlopen(req) as response:
            status_code = response.getcode()
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            
            print(f"✅ 收到容器响应 (HTTP {status_code}): {res_body}")
            if "emotion" not in res_json:
                print("❌ 验证异常：容器返回体错误！")
                cleanup()
                sys.exit(1)
            print("🎉 容器内网络连通性验证通过，API 工作正常！")

    except urllib.error.URLError as e:
        print(f"❌ 容器请求失败: {e}")
        cleanup()
        sys.exit(1)
        
    cleanup()

def cleanup():
    print("🧹 正在清理拆卸 Docker 容器...")
    subprocess.run("docker-compose down", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if os.path.exists("docker-compose.override.yml"):
        os.remove("docker-compose.override.yml")
    print("🏁 Docker 清理完成！测试结束。")

if __name__ == "__main__":
    main()
