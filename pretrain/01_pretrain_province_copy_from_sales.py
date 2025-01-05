"""最原始训练循环，简单模式"""

import requests
import torch
import os
import tiktoken
from transformers import AutoTokenizer

from model_define.configuration_futureai import FutureAiConfig
from model_define.modeling_futureai import FutureAiModel

# 参数定义
batch_size = 2  # How many batches per training step
context_length = 16  # Length of the token chunk each batch
learning_rate = 1e-3  # 0.001
dropout = 0.1  # Dropout rate
max_iters = 500  # Total of training iterations <- Change this to smaller number for testing
eval_interval = 50  # How often to evaluate
eval_iters = 20  # Number of iterations to average for evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

with open('../dataset_file/dataset_province/province_city.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 使用 Qwen Tokenzier 对文本进行编码
tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")
tokenized_text = tokenizer.encode(text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)  # put tokenized text into tensor

# 拆分训练数据与评估数据
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]

# 模型初始化
config = FutureAiConfig()
config.dropout = dropout
model = FutureAiModel(config)
model = model.to(device)

# 获取训练数据
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y


# 计算损失函数
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            inference_res = model(x_batch, y_batch)
            logits = inference_res.logits
            loss = inference_res.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Use AdamW optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()

# 训练循环
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    print(xb)
    inference_res = model(xb, yb)
    logits = inference_res.logits
    loss = inference_res.loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 模型保存
torch.save(model.state_dict(), '../checkpoint_folder/checkpoint_province/model-province.pt')

# Generate
model.eval()
for i in range(10):
    start = '江苏'
    start_ids = tokenizer.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    y = model.generate(x, max_new_tokens=5)
    print('---------------')
    print(tokenizer.decode(y[0].tolist()))
    print('---------------')