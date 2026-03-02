# 🎭 细粒度情绪识别系统 (Fine-grained Emotion Recognition API)

本项目基于 **Qwen2.5-7B-Instruct** 并结合团队微调的 **LoRA 权重 (`fQwQf/erc-qwen2.5-7b-sota`)**，实现了一个支持多轮上下文理解的轻量级细粒度情绪识别接口。

您可以输入单句发言，或是提供整段对话历史，系统将推断出说话者的角色、受前文影响的程度，并在 28 种极细粒度的情感分类（如 `anger`、`joy`、`sadness`、`anticipation` 等）中做出精准预测。

---

## 🏆 模型性能 (SOTA Metrics)

我们的模型在多轮对话情绪识别（ERC）任务中达到了业界领先的水平 (State-of-the-Art)。相比于仅针对单句进行分类的传统模型，本系统突破性地引入了 **Context-Aware（上下文感知）** 机制。

*   **EmpatheticDialogues 数据集评估**
    *   **Accuracy (准确率)**: 显著优于 baseline 模型，在庞杂的 32 类/28 类细粒度情绪中表现出极强的辨别能力。
    *   **上下文推理提取**: 模型被专门教导关注 `Impact`，即“前文历史是如何促成了说话者此刻的情绪”，这极大地降低了单句歧义带来的误判。
*   **指令遵从与推理**
    作为 7B 级别的生成式底座模型，它不单单输出一个概率标签，而是直接像人类一样进行结构化思考，输出 `Emotion`、`Speaker` 和 `Impact`。

---

## ⚡ 核心特性

- **一键极速版测试环境**：附带了一个无需下载 15GB 模型的纯前端互动测试脚本 `test_main.py`。
- **动态硬件调度**：优先检测 NVIDIA GPU 并自动开启 4-bit 量化（节约显存）；若无显卡，自动丝滑降级到纯 CPU 运行。
- **现代化互动前端**：自带黑客帝国风格、流畅沉浸的 Dark Mode 分析网页，可供演示与直观验证。
- **Docker 持久化部署**：支持 Docker Compose 容器化一键部署，且缓存目录外置，防止重复下载大模型。

---

## 🛠️ 快速上手指南

### 方式一：纯本地 Python 部署

**1. 克隆代码库**
```bash
git clone https://github.com/EmotionAI-2026-WHU/emotion-recognition
cd emotion-recognition/Emotion_Ai_Api_CPU
```

**2. 安装依赖环境**
```bash
# 推荐使用 Conda 虚拟环境
pip install -r requirements.txt
```

**3. 启动 API 服务**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
启动后，第一次请求会自动从 HuggingFace 官方镜像加速下载大模型底座和我们将其实例化的专属 LoRA 权重。打开 `http://localhost:8000` 即可欣赏 UI 界面。

### 方式二：Docker Compose 一键容器化 (推荐)

如果你不想弄乱本地的 Python 环境，只需要确保安装了 Docker：

```bash
cd Emotion_Ai_Api_CPU
docker-compose up --build -d
```
Docker 会自动探测你的 NVIDIA GPU；如无，则作为 CPU 服务运行。模型会固化在 Docker Volume 里，随意重启不会丢失。

---

## 🚀 API 接口参考

您可以将这套系统接入自己的前端或 APP，我们提供了标准的 RESTful 接口。

**端点**: `POST /predict`

**请求参数 (JSON)**
```json
{
  "text": "我都走到半路了，突然下大暴雨，全身都淋湿了！",
  "history": ["今天天气预报明明说没雨的", "是啊，所以我没带伞"]
}
```

**返回体 (JSON)**
```json
{
  "emotion": "annoyance",
  "speaker": "Speaker",
  "impact": "被错误的天气预报误导导致淋雨"
}
```

---
*Powered by WHU EmotionAI 2026 Team.*
