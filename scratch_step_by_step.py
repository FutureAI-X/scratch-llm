import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import tiktoken
import pandas as pd

"""模型参数定义"""
# 批次大小
batch_size=4
# 每一批训练数据的token个数
context_length = 16
# 模型的维度
d_model=64
# 注意力头数
num_heads = 4

"""获取训练数据"""
if not os.path.exists("data/sales_textbook.txt"):
    url = "https://hf-mirror.com/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt"
    with open('data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

with open("data/sales_textbook.txt", "r", encoding="utf-8") as f:
    text = f.read()

""""tokenize text"""
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)
max_token_value = tokenized_text.max().item()

"""split text to train and test"""
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]
valid_data = tokenized_text[split_idx:]

"""准备训练数据"""
data = train_data
idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
x_batch = torch.stack([torch.tensor(data[i:i+context_length]) for i in idxs])
y_batch = torch.stack([torch.tensor(data[i+1:i+context_length+1]) for i in idxs])


"""定义Token Embedding表格"""
token_embedding_lookup_table = nn.Embedding(max_token_value+1, d_model)
x_batch_embeddings = token_embedding_lookup_table(x_batch)
y_batch_embeddings = token_embedding_lookup_table(y_batch)

"""处理位置编码"""
position_encoding_lookup_table = torch.zeros(context_length, d_model)
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size, -1, -1)

"""输入向量与位置编码相加"""
input_embeddings_x = x_batch_embeddings + position_encoding_lookup_table
input_embeddings_y = y_batch_embeddings + position_encoding_lookup_table

"""准备QKV"""
X = input_embeddings_x
Wq = nn.Linear(d_model, d_model)
Wk = nn.Linear(d_model, d_model)
Wv = nn.Linear(d_model, d_model)

Q = Wq(X)
K = Wk(X)
V = Wv(X)

Q = Q.view(batch_size, context_length, num_heads, d_model//num_heads).transpose(1, 2)
K = K.view(batch_size, context_length, num_heads, d_model//num_heads).transpose(1, 2)
V = V.view(batch_size, context_length, num_heads, d_model//num_heads).transpose(1, 2)

"""注意力计算"""
attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model // num_heads)
mask = torch.triu(torch.ones(attention_score.shape[-2:]), diagonal=1).bool()
attention_score = attention_score.masked_fill(mask, -float('inf'))

attention_score = F.softmax(attention_score, dim=-1)

"""@V"""
A = torch.matmul(attention_score, V)
A = A.transpose(1, 2).contiguous().view(batch_size, context_length, d_model)

Wo = nn.Linear(d_model, d_model)
output = Wo(A)

output = output + X

"""Layer Norm"""
layer_norm = nn.LayerNorm(d_model)
output_layer_norm = layer_norm(output)

"""前馈网络"""
output = nn.Linear(d_model, 4 * d_model)(output_layer_norm)
output = nn.ReLU()(output)
output = nn.Linear(4 * d_model, d_model)(output)

output = output + output_layer_norm

output = layer_norm(output)

"""计算输出"""
logits = nn.Linear(d_model, max_token_value+1)(output)

probabilities = torch.softmax(logits, dim=-1)

pd.DataFrame(probabilities[0].detach().numpy()).plot(kind='bar')

"""
def get_batch(split):
    data = train_data if split == 'train' else valid_data
    ix = torch.randint(len(data) - content_length, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+content_length]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+content_length+1]) for i in ix])
    return x, y
"""