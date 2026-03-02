#!/bin/bash
# 细粒度情绪识别项目 - 端到端工程管线脚本 (Linux / AutoDL 适用)
# 设置遇到错误立即停止执行
set -e

echo "🚀 开始情绪识别 AI 训练管线..."

# ==========================================
# 阶段 1: 环境配置
# ==========================================
echo -e "\n[1/4] 配置 Python 依赖环境..."
# 检查当前目录下是否有 requirements.txt，如果没有则尝试进入 ERC 目录
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
elif [ -f "Emotion_Ai_Api_CPU/requirements.txt" ]; then
    pip install -r Emotion_Ai_Api_CPU/requirements.txt
else
    echo "未找到 requirements.txt，安装核心依赖..."
    pip install torch==2.1.0 transformers peft accelerate bitsandbytes==0.41.1 fastapi uvicorn pydantic
fi

# ==========================================
# 阶段 2: 数据下载 (源自 HuggingFace)
# ==========================================
echo -e "\n[2/4] 连接镜像节点并下载原始数据集..."
# 针对 AutoDL 或国内云服务器配置 HuggingFace 镜像加速
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER="1"

# 确保目录存在
mkdir -p data/raw
mkdir -p cache/data
mkdir -p outputs/my_finetuned_lora

# 下载原始数据
echo ">>> 执行 organize_raw_datasets.py..."
python ERC/scripts/organize_raw_datasets.py

# ==========================================
# 阶段 3: 数据清洗与特征预处理
# ==========================================
echo -e "\n[3/4] 清洗数据并构建 Context / 对话级 Sample 缓存..."
echo ">>> 执行 prepare_data.py..."
python ERC/scripts/prepare_data.py

# ==========================================
# 阶段 4: 模型底层下载与并行微调训练
# ==========================================
echo -e "\n[4/4] 启动 Qwen2.5-7B LoRA 微调训练..."
echo ">>> 执行 train_7b_sota.py..."
# 此脚本在运行时遇到 AutoModelForCausalLM.from_pretrained 会自动从镜像下载基座模型
# 然后运用 QLoRA 进行 4-bit 并行训练
python ERC/scripts/train_7b_sota.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "./outputs/my_finetuned_lora" \
    --epochs 3 \
    --batch_size 1 \
    --grad_acc 32 \
    --lr 2e-5

echo -e "\n✅ 端到端管线执行完毕！"
echo "🎉 恭喜！您微调后的 LoRA 权重已保存在 './outputs/my_finetuned_lora' 目录中。"
echo "💡 接下来您只需将 Emotion_Ai_Api_CPU/api/model_inference.py 中的 LORA_PATH 修改为上面这个目录，然后执行 uvicorn 命令即可部署您的 API！"
