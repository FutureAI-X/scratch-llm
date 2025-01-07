"""
自定义 Transformer 模型预训练
模型：自定义 Transformer 模型
数据集：省份介绍/省会数据
方式：使用 Hugging Face 的 trainer 类
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer

from model_define.model_qwen.configuration_qwen2 import Qwen2Config
from model_define.model_qwen.modeling_qwen2 import Qwen2ForCausalLM


"""
MODULE 0 定义训练任务
city：省会信息
else：省份介绍
"""
dataset_type = "city"
dataset_file_name = "province_city" if dataset_type == "city" else "province_intro"
model_save_path = "qwen2-province-city" if dataset_type == "city" else "qwen2-province-intro"


"""
MODULE 1 定义训练参数
learning_rate：学习率
epochs：循环次数
device：设备（CUDA or CPU）
"""
learning_rate = 1e-3
epochs = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'


"""
MODULE 2 Tokenizer 与 Model 定义
"""
# 2.1 初始化 Tokenizer（使用 Qwen 2.5 的 Tokenizer）
tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")

# 2.2初始化模型参数
config = Qwen2Config()
config.max_position_embeddings = 32
config.hidden_size = 64
config.num_attention_heads = 8
config.num_key_value_heads = 2
config.num_hidden_layers = 4

# 2.3 初始化模型
model = Qwen2ForCausalLM(config)
model = model.to(device)


"""
MODULE 3 数据集加载与Token化
"""
# 3.1 加载数据集
dataset = load_dataset('json', data_files={"train": f"../dataset_file/dataset_province/{dataset_file_name}.jsonl"})

# 3.2 定义数据集[token化]处理函数
def tokenize_function(batch):
    """
    对原始文本进行token化，生成新的参数：
        input_ids       token化之后的数值
        attention_mask  损失掩码，为1表示需要计算损失，为0表示不计算损失
    """
    tokenizer_temp = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=config.max_position_embeddings + 1)
    tokenizer_temp["labels"] = tokenizer_temp["input_ids"].copy()
    return tokenizer_temp

# 3.3 执行数据集[token化]处理
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


"""
MODULE 4 定义训练参数
"""
training_args = TrainingArguments(
    output_dir=f"../checkpoint_folder/checkpoint_province/{model_save_path}",
    overwrite_output_dir=True,  # 如果设置为 True，则允许覆盖输出目录中的现有内容。
    num_train_epochs=500,  # 训练的总轮数（epoch），即整个数据集会被遍历 50 次。
    per_device_train_batch_size=10,  # 每个设备（GPU/CPU）上的训练批次大小。如果使用多个 GPU，总批次大小将是这个值乘以 GPU 数量。
    save_steps=100,  # 每多少步保存一次模型检查点。这里表示每 50 步保存一次。
    save_total_limit=3,  # 最多保存多少个检查点。当超过这个限制时，最早的检查点将被删除。
    logging_steps=1,  # 每多少步记录一次日志。这里表示每 1 步记录一次。
    learning_rate=1e-3,  # 学习率，用于优化器更新模型参数的速度。这里设置为 0.001。
    weight_decay=0.01,  # 权重衰减（L2正则化）系数，用于防止过拟合。
)


"""
MODULE 5 定义训练器并执行训练
"""
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)

trainer.train()
