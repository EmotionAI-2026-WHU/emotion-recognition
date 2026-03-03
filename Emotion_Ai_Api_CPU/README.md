# 细粒度对话情绪识别 API

本项目基于 **Qwen2.5-7B-Instruct** 基座模型和经过微调的 **LoRA 权重**，提供了一款轻量级的情绪识别 API。您只需发送一段对话文本，即可获得对应的细粒度情绪标签（如 `anger`、`joy`、`sadness` 等）。项目采用 **FastAPI** 构建，支持高并发请求，并已通过 CPU 环境验证，可快速部署使用。

## 技术栈
- **编程语言**：Python 3.10
- **环境管理**：Miniconda3
- **Web 框架**：FastAPI + Uvicorn
- **深度学习框架**：PyTorch 2.1
- **模型加载**：Transformers + PEFT
- **依赖管理**：pip + requirements.txt

## 项目依赖
### 核心库
```
#requirements.txt
# 核心依赖
torch==2.1.0
transformers
peft
accelerate
bitsandbytes==0.41.1  # 用于4-bit量化

# Web框架
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart  # 处理表单数据

# 工具
rich  # TUI中用到的美化库（虽然API不需要，但解析函数可能依赖）
```

## 使用方法
### 1 环境配置
**先创建conda环境**（miniconda），
然后下载依赖项：在项目根目录执行
```
pip install -r requirements.txt
```
下载所需工具。

### 2 下载LoRA权重
该api需要 **Qwen2.5-7B-Instruct 基座模型**  和  **ERC 微调 LoRA 权重**。运行 download_models.py（可在文件中自定义下载路径），将LoRA权重下载到本地。

对于基座模型，第一次运行model_inference.py 的 load_model()函数时会自动下载基座模型到缓存。基座模型大约15G，因此确保C盘留有充足空间。当然，如果本地已经下载好基座模型，可将 model_inference.py 中的 **BASE_MODEL_NAME** 改为 基座模型所在的本地文件夹路径
### 3 设置本地文件路径
在基座和LoRA下载好之后，修改 model_inference.py 中的路径。

**LORA_PATH** 改为 LoRA模型所在的本地文件夹路径（例如 "D:/models/lora"）。

可选：**BASE_MODEL_NAME** 改为 基座模型所在的本地文件夹路径（例如 "D:/models/qwen"）。
例：如果**BASE_MODEL_NAME**改为了D:/models/qwen，请确保目录结构如下：
```
D:/models/qwen/
  ├── model-00001-of-00004.safetensors
  ├── model-00002-of-00004.safetensors
  ├── model-00003-of-00004.safetensors
  ├── model-00004-of-00004.safetensors
  ├── config.json
  └── ...
```
### 4 启动API服务
在项目根目录执行
```
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
打开 http://127.0.0.1:8000/docs ，在 Swagger UI 中直接测试 /predict 接口。

## 常见问题
#### Q：基座模型加载不到100%就崩了
#### A：可以尝试设置虚拟内存。

#### Q：predict好慢
#### A：因为现在是100%依赖cpu而没用gpu，所以一次predict要等几分钟

## 本项目结构
```
EmotionAiApi/
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── model_inference.py
├── src/
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       └── prompt_template.py
├── requirements.txt
├── README.md
└── download_models.py      
```