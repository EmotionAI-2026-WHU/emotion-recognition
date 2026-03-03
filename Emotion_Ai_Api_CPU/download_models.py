from huggingface_hub import snapshot_download
'''
# 下载基座模型
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="D:/models/qwen",
    local_dir_use_symlinks=False,
    resume_download=True
)
'''
# 下载 LoRA 权重
snapshot_download(
    repo_id="fQwQf/erc-qwen2.5-7b-sota",
    local_dir="D:/models/lora",
    local_dir_use_symlinks=False,
    resume_download=True
)