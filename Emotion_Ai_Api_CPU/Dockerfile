# 基于官方 Python 3.10 轻量级镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # 国内构建加速配置：设定 HF 镜像，防止下模型下半天超时
    HF_ENDPOINT="https://hf-mirror.com"

# 安装系统编译所需的基础库（如部分 Python 库需要 c 文件编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装支持 CUDA 的 PyTorch 2.1 (虽然体积较大，但可以跑满 GPU 算力；如果没有 GPU 会自动降级为 CPU 计算)
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 复制依赖描述文件并安装项目其余依赖
# 这里我们过滤并替换掉原 requirements.txt 中的 torch==2.1.0 (因为上一步指定了 cpu source 安装)
COPY requirements.txt .
RUN grep -v "torch==" requirements.txt > req_no_torch.txt \
    && pip install --no-cache-dir -r req_no_torch.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 把所有代码搬入
COPY . .

# 暴露 FastAPI 的端口
EXPOSE 8000

# 启动 API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
