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

### 端到端自动化管线 (`run_pipeline.sh`)

本项目根目录下提供了一个 `run_pipeline.sh` 脚本，其核心作用是**提供一键式、端到端的模型拉取与 LoRA 微调 (SFT) 工作流**。主要针对需要从头训练或在 AutoDL 等云服务器上快速复现实验的开发者。
执行该脚本 (`bash run_pipeline.sh`) 将自动串行触发以下全过程：
1. **环境拦截与安装**：检测系统并强行补齐 `transformers`, `peft`, `bitsandbytes` 等核心训练依赖。
2. **连接 HuggingFace 镜像节点**：配置 `HF_ENDPOINT` 到国内云镜像加速下载。
3. **数据集处理**：先后调用 `organize_raw_datasets.py` 和 `prepare_data.py` 将原始文本流格式化为含蓄的对话大背景 Prompt Context 片段，建立缓存。
4. **Qwen2.5 启动并行微调**：命令底层引擎开始拉取基础底座模型并发起 `train_7b_sota.py`，采用 4-bit 并行策略跑完几个 Epoch，微调后的个人适配器（Adapter）权重将被导出至 `./outputs/my_finetuned_lora` 中，供后续 API 直接挂载使用。

## 性能指标 (State-of-the-Art)

模型在对话情绪识别（Emotion Recognition in Conversations, ERC）的标准化评估中达到 State-of-the-Art (SOTA) 级别的性能。

- **极细粒度标签识别极限 (Fine-Grained ERC)**：在 GoEmotions 测试集的 28 类极细粒度情绪空间中，本模型超越了传统专用型网络结构，取得了 **53.55%** 的 Weighted F1 成绩。
- **高阶情绪价态聚类 (Coarse-Grained Upper Bound)**：即使面对高度复杂的人际对话，当对标签进行情绪价态（8 分类 Valence 簇）聚合时，模型性能依然表现出了高度的稳定识别力，精度达到了卓越的 **73.38%** Weighted F1。
- **稳健的上下文感知 (Robust Context Awareness)**：基于生成式大语言模型底座强劲的信息融合机制，本模型能有效提取长序列对话历史的时序语义，规避了传统孤立短文本分类带来的短视与逻辑误判。

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
  "emotion": "annoyance"
}
```

---
*Powered by WHU EmotionAI 2026 Team.*
