import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer

from model_define.model_qwen.configuration_qwen2 import Qwen2Config
from model_define.model_qwen.modeling_qwen2 import Qwen2ForCausalLM

learning_rate = 1e-3
epochs = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")

# 定义Model
config = Qwen2Config()
config.max_position_embeddings = 32

config.hidden_size = 64
config.num_attention_heads = 8
config.num_key_value_heads = 2
config.num_hidden_layers = 4
model = Qwen2ForCausalLM(config)
model = model.to(device)

dataset = load_dataset('json', data_files={"train": "../dataset_file/dataset_province/province_city.jsonl"})

def tokenize_function(batch):
    """对原始文本进行token化，生成新的参数：
        input_ids       token化之后的数值
        attention_mask  损失掩码，为1表示需要计算损失，为0表示不计算损失
    """
    tokenizer_temp = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=config.max_position_embeddings + 1)
    tokenizer_temp["labels"] = tokenizer_temp["input_ids"].copy()
    return tokenizer_temp

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(tokenized_datasets["train"][1])

training_args = TrainingArguments(
    output_dir="../checkpoint_folder/checkpoint_province/qwen2-source",
    overwrite_output_dir=True,  # 如果设置为 True，则允许覆盖输出目录中的现有内容。
    num_train_epochs=50,  # 训练的总轮数（epoch），即整个数据集会被遍历 3 次。
    per_device_train_batch_size=1,  # 每个设备（GPU/CPU）上的训练批次大小。如果使用多个 GPU，总批次大小将是这个值乘以 GPU 数量。
    save_steps=50,  # 每多少步保存一次模型检查点。这里表示每 10,000 步保存一次。
    save_total_limit=3,  # 最多保存多少个检查点。当超过这个限制时，最早的检查点将被删除。
    logging_steps=1,  # 每多少步记录一次日志。这里表示每 500 步记录一次。
    learning_rate=1e-3,  # 学习率，用于优化器更新模型参数的速度。这里设置为 0.00002。
    weight_decay=0.01,  # 权重衰减（L2正则化）系数，用于防止过拟合。
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)

trainer.train()

# {'loss': 0.0295, 'grad_norm': 0.20351974666118622, 'learning_rate': 0.0, 'epoch': 50.0}
