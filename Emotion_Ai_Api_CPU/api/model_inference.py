
#给出上面两个函数的说明
"""
load_model(): 加载基础模型和 LoRA 权重，并初始化提示构建器。根据配置选择是否使用量化。
predict(text, history): 接收输入文本和对话历史，构建提示并生成模型输出。解析输出以提取情绪、说话人和影响。
""" 
# api/model_inference.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.prompt_template import EmotionPromptBuilder, parse_model_output

# 配置
#BASE_MODEL_NAME = "C:/Users/biggboss01/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"#"Qwen/Qwen2.5-7B-Instruct"是说，如果本地缓存有，自动加载；没有会从 Hugging Face 下载
LORA_PATH = "D:/models/lora"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_QUANTIZE = True if DEVICE == "cuda" else False # 如果是CUDA环境默认开启4-bit节约显存

model = None
tokenizer = None
builder = None

def load_model():
    global model, tokenizer, builder
    print(f"Selecting device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        print(f"Attempting to load full model on {DEVICE}...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto" if DEVICE == "cuda" else None
        )
        if DEVICE == "cpu":
            model.to("cpu")
        print("Full model loaded successfully.")
    except Exception as e:
        print(f"Failed to load full model: {e}")
        if DEVICE == "cuda":
            print("Attempting to load model with 4-bit quantization as fallback...")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            print("4-bit quantized model loaded successfully.")
        else:
            print("Cannot fallback to 4-bit quantization on CPU.")
            raise e

    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()

    builder = EmotionPromptBuilder(use_retrieval=False)
    print("Model loaded successfully.")

def predict(text: str, history=None):
    if history is None:
        history = []
    prompt = builder.build_inference_prompt(history, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    analysis = parse_model_output(gen_text)
    return analysis

