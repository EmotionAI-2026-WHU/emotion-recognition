# Fine-grained Emotion Recognition API

本项目提供了一个支持多轮对话上下文理解的细粒度和轻量化情绪识别系统。基于 **Qwen2.5-7B-Instruct** 作为底座模型，并结合 PEFT/LoRA 参数高效微调技术（权重标示：`fQwQf/erc-qwen2.5-7b-sota`），本系统致力于在 28 种细粒度情绪分类集合中，实现高精度的情感辨识与上下文意图提取。

## 项目特性

- **上下文感知计算 (Context-Aware Reasoning)**：突破传统单句情绪分类的局限，深度融合对话历史以抽取上下文情境信息，显著降低孤立文本带来的语义歧义。
- **硬件自适应调度 (Dynamic Hardware Scheduling)**：系统内置动态设备探针。在具备 NVIDIA 硬件的环境下自动开启 GPU 加速及 4-bit 量化（基于 BitsAndBytes），以优化显存占用；在受限环境下则平稳降级至纯 CPU 计算模式。
- **交互式 Web 界面 (Interactive Web Interface)**：内置一套遵循现代 Web 设计准则构建的响应式前端面板，支持实时推理过程与结构化分析结果的可视化呈现。
- **标准化容器部署 (Containerized Deployment)**：提供生产级别的 Dockerfile 与 Docker Compose 编排方案，支持跨平台一键部署与持久化外部挂载，避免大模型的重复下载损耗。
- **环境隔离验证机制 (Zero-Dependency Mock Environment)**：集成独立的纯前端联调脚本 (`test_main.py`)，允许在不加载庞大底层 LLM 权重的状态下进行接口及前端的快速验证。

## 项目结构

```text
.
├── api/                
│   ├── main.py              # FastAPI 接口实现与路由定义
│   ├── model_inference.py   # 模型加载、量化配置及推理核心算法逻辑
│   ├── test_main.py         # 脱离模型依赖的纯前端模拟与联调脚本
│   └── static/              # 交互式 Web 界面视觉与前端逻辑资产
├── Dockerfile               # 容器化镜像构建规范配置
├── docker-compose.yml       # 多节点与外置存储空间服务编排方案
└── run_pipeline.sh          # 端到端自动化数据预处理与监督微调执行脚本
```

## 模型训练工作流

1. **数据预处理**：基于 EmpatheticDialogues 与 GoEmotions 等开源语料库，通过上下文拼接算法构建多轮对话样本，强化模型对时序历史信息的感知能力。
2. **有监督微调 (SFT)**：采用 Qwen2.5-7B-Instruct 生成式语言模型作为计算基座，利用 PEFT/LoRA 技术在情绪识别这一垂直特定任务上进行参数高效微调。
3. **推理优化**：深度集成 BitsAndBytes 4-bit 数值量化方案，在维持原始模型表征精度的同时，严控显存分配开销并提升端到端吞吐率。
4. **服务化封装**：基于 ASGI 框架 FastAPI 构建高度异步化与非阻塞的对内/对外接口，实现底层模型计算节点与上层业务前端的高效融合。

## 性能指标 (State-of-the-Art)

模型在对话情绪识别（Emotion Recognition in Conversations, ERC）的标准化评估中达到 State-of-the-Art (SOTA) 级别的性能。

- **泛化表现与精确度验证**：在 EmpatheticDialogues 基准测试中，本模型的精度超越了传统专用型网络结构，在极为细化的标签分布（多达 28 类情绪空间）中体现出稳健的模式识别精度。
- **因果联系提取 (Impact Extraction)**：在训练范式设计中，模型被施加强迫注意力以探究“影响变量 (Impact)”——即分析前序对话事件如何诱发当事人的即时情绪变化。该机制显著消除了缺乏背景推导的短文本误判问题。
- **系统提示词服从能力**：作为 70 亿参数量级的逻辑生成基础底座，模型跳出了传统分类器仅反馈固定标签向量的藩篱，能依据人类指令结构化生成 `Emotion`（情感判定）、`Speaker`（发言者角色识别）与 `Impact`（因果归因），实现具备深度解释性的白盒预判过程。

## 部署与使用方法

### 方案一：Docker 容器化部署 (推荐)

借助 Docker 提供的一致性运行环境，服务可自动分配底层加速硬件并实现数据隔离。

1. **拉取项目库**：
   ```bash
   git clone https://github.com/EmotionAI-2026-WHU/emotion-recognition.git
   cd emotion-recognition/Emotion_Ai_Api_CPU
   ```

2. **编排并启动服务**：
   ```bash
   docker-compose up --build -d
   ```
   *说明：Docker Compose 引擎将自动接管 NVIDIA GPU 映射，并在本地宿主机创建卷（Volume）以托管庞大的基座模型缓冲文件，确保容器在历经销毁与重建时免于发起冗余的网络下载请求。*

### 方案二：本地原生 Python 环境

1. **拉取项目库**：
   ```bash
   git clone https://github.com/EmotionAI-2026-WHU/emotion-recognition.git
   cd emotion-recognition/Emotion_Ai_Api_CPU
   ```

2. **解析并装载依赖** (建议在 Conda 等独立虚拟环境中执行)：
   ```bash
   pip install -r requirements.txt
   ```

3. **激活 API 服务节点**：
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   *程序初次挂载时，内置脚本将连接至 Hugging Face Hub (或配置镜像源) 完成底层张量文件的按需热下载与内存预分配。*

服务部署完毕后，访问 `http://localhost:8000` 即可加载图形化的前端分析界面，前往 `http://localhost:8000/docs` 查阅 OpenAPI / Swagger 标准化的交互式测试文档。

## 核心接口说明 (API Reference)

**访问端点**: `POST /predict`

**入参规范 (Request Payload - JSON格式)**:
```json
{
  "text": "我都走到半路了，突然下大暴雨，全身都淋湿了！",
  "history": [
    "今天天气预报明明说没雨的",
    "是啊，所以我没带伞"
  ]
}
```

**出参标准 (Response Payload - JSON格式)**:
```json
{
  "emotion": "annoyance",
  "speaker": "Speaker",
  "impact": "被错误的天气预报误导导致突发的生活困扰"
}
```

---
*Powered by WHU EmotionAI 2026 Team.*
