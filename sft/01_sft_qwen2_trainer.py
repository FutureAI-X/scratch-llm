"""
MODULE 0 定义任务类型
city：省会信息
else：省份介绍
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from model_define.model_qwen.configuration_qwen2 import Qwen2Config
from model_define.model_qwen.modeling_qwen2 import Qwen2ForCausalLM

dataset_file_name = "sft_demo"
model_save_name = "qwen2-sft-qa"

"""
MODULE 1 定义运行设备
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
MODULE 2 Tokenizer 与 Model 定义
"""
# 2.1 初始化 Tokenizer（使用 Qwen 2.5 的 Tokenizer）
tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")

checkpoint = ""

if checkpoint:
    model = Qwen2ForCausalLM.from_pretrained(checkpoint)
else:
    # 2.2初始化模型参数
    config = Qwen2Config()
    config.max_position_embeddings = 100
    config.hidden_size = 64
    config.num_attention_heads = 8
    config.num_key_value_heads = 8
    config.num_hidden_layers = 4

    # 2.3 初始化模型
    model = Qwen2ForCausalLM(config)
    model = model.to(device)

"""
MODULE 3 数据集加载与Token化
"""
# 3.1 加载数据集
dataset = load_dataset('json', data_files={"train": f"../dataset_file/sft/{dataset_file_name}.jsonl"})

sft_config = SFTConfig(
    output_dir=f"../checkpoint_folder/sft/{model_save_name}",
    overwrite_output_dir=True,
    max_steps=5000,  # Adjust based on dataset size and desired training duration
    per_device_train_batch_size=2,  # Set according to your GPU memory capacity
    learning_rate=1e-3,  # Common starting point for fine-tuning
    logging_steps=50,  # Frequency of logging training metrics
    save_steps=500,  # Frequency of saving model checkpoints
    save_total_limit=3,
    use_mps_device=(
        True if device == "mps" else False
    ),  # Use MPS for mixed precision training
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer
)
trainer.train()