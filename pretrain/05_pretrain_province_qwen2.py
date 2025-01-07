"""
Qwen2 模型预训练
模型：Qwen2 模型
数据集：省份介绍/省会数据
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model_define.model_qwen.configuration_qwen2 import Qwen2Config
from model_define.model_qwen.modeling_qwen2 import Qwen2ForCausalLM


"""
MODULE 0 定义训练任务
city：省会信息
else：省份介绍
"""
dataset_type = "city"
dataset_file_name = "province_city" if dataset_type == "city" else "province_intro"
model_save_name = "qwen2-province-city" if dataset_type == "city" else "qwen2-province-intro"


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
dataset = load_dataset('json', data_files=f"../dataset_file/dataset_province/{dataset_file_name}.jsonl", split='train')

# 3.2 定义数据集[token化]处理函数
def tokenize_function(batch):
    """
    对原始文本进行token化：
        input_ids       token化之后的数值
        attention_mask  损失掩码，为1表示需要计算损失，为0表示不计算损失
    """
    tokenizer_temp = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=config.max_position_embeddings + 1)
    tokenizer_temp["labels"] = tokenizer_temp["input_ids"].copy()
    return tokenizer_temp

# 3.3 执行数据集[token化]处理
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


"""
MODULE 4 数据集转换
将Token化之后的数据集转为PyTorch的DataLoader
"""
# 4.1 定义自定义的collate_fn
def custom_collate_fn(batch):
    """
        默认的数据集格式与模型期望的输入有所差别，使用此函数进行转换
        input_ids：原始文本token化之后的结构
        attention_mask：注意力掩码
        labels：期望的输出，用于计算损失函数
        """
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch]).to(device)
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch]).to(device)
    labels = torch.stack([torch.tensor(item['labels']) for item in batch]).to(device)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# 4.2 定义PyTorch的DataLoader
train_dataloader = DataLoader(
    tokenized_datasets,
    batch_size=2,
    shuffle=False,
    collate_fn=custom_collate_fn
)

train_dataloader_lengt = len(train_dataloader)
print(f"每轮循环的训练步骤为：{train_dataloader_lengt}")


"""
MODULE 5 执行训练
"""
# 5.1 定义优化器：优化器用于更新模型参数、控制学习率
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 5.2 定义训练循环
for epoch in range(epochs):
    # 变量i用于记录单个epoch内训练的步数
    i = 1
    for batch in train_dataloader:
        # 执行模型预测
        outputs = model(**batch)

        # 获取损失函数
        loss = outputs.loss

        print(f"Epoch {epoch}, Loss: {loss.item()}, Process: {i}/{train_dataloader_lengt}")

        # 计算损失函数对模型参数的梯度：通过调用 backward() 方法，PyTorch 会自动计算损失函数相对于每个模型参数的梯度，并将这些梯度存储在各个参数的 .grad 属性中
        loss.backward()

        # 使用优化器更新模型参数：根据之前计算得到的梯度，优化器（如AdamW）会按照设定的学习率和优化算法更新模型参数，以最小化损失函数
        optimizer.step()

        # 清空梯度：为了防止梯度累积，在每次参数更新后需要将所有参数的梯度清零。set_to_none=True 是一种更高效的清零方式，它将梯度设置为 None 而不是显式地将其设为零张量，从而节省内存
        optimizer.zero_grad(set_to_none=True)

        # 记录训练步骤
        i = i + 1

        # 每100步保存一次模型
        if i % 100 == 0:
            torch.save(model.state_dict(), f'../checkpoint_folder/checkpoint_province/{model_save_name}.pt')

    # 每轮循环结束，保存模型
    torch.save(model.state_dict(), f'../checkpoint_folder/checkpoint_province/{model_save_name}.pt')