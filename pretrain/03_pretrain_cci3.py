import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding, Trainer

from model_define.model_futureai.configuration_futureai import FutureAiConfig
from model_define.model_futureai.modeling_futureai import FutureAiModel

# 定义运行设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_file/tokenizer_qwen")

print(tokenizer.vocab_size)

# 定义Model
config = FutureAiConfig()
model = FutureAiModel(config)
model = model.to(device)

# 加载数据集
dataset = load_dataset("json", data_dir="./dataset_file/cci3_hq_simple", split="train")

# 对原始数据进行Token化
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)

# 定义数据集格式
def prepare_dataset(examples):
    input_ids = examples['input_ids']
    attention_mask = examples['attention_mask']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids  # 自监督学习，标签与输入相同
    }

dataset = dataset.map(prepare_dataset, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="../checkpoint_folder", # 模型输出目录，训练结果（如检查点、最终模型等）将保存在这个文件夹中。
    overwrite_output_dir=True, # 如果设置为 True，则允许覆盖输出目录中的现有内容。
    num_train_epochs=5, # 训练的总轮数（epoch），即整个数据集会被遍历 3 次。
    per_device_train_batch_size=1, # 每个设备（GPU/CPU）上的训练批次大小。如果使用多个 GPU，总批次大小将是这个值乘以 GPU 数量。
    save_steps=500, # 每多少步保存一次模型检查点。这里表示每 10,000 步保存一次。
    save_total_limit=3, # 最多保存多少个检查点。当超过这个限制时，最早的检查点将被删除。
    logging_dir="./train_log", # 日志输出目录，训练日志将保存在这个文件夹中。
    logging_steps=1, # 每多少步记录一次日志。这里表示每 500 步记录一次。
    evaluation_strategy="epoch", # 评估策略，"epoch" 表示在每个 epoch 结束时进行评估。
    learning_rate=3e-4, # 学习率，用于优化器更新模型参数的速度。这里设置为 0.00002。
    weight_decay=0.01, # 权重衰减（L2正则化）系数，用于防止过拟合。
)

# 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}

# 定义数据收集器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # 使用同一数据集进行评估（示例）
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()