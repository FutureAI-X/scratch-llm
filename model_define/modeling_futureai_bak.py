import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from model_define.configuration_futureai import FutureAiConfig

class FeedForward(nn.Module):
    """定义前馈神经网络"""
    def __init__(self, config: FutureAiConfig):
        super().__init__()
        self.d_model = config.d_model
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.ffn(x)

class Attention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, config: FutureAiConfig):
        super().__init__()

        # 定义常规变量
        self.d_model = config.d_model
        self.head_size = config.head_size
        self.context_length = config.context_length

        # 定义Wq、Wk、Wv
        self.Wq = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.Wk = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.Wv = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)

        # 定义 dropout
        self.dropout_layer = nn.Dropout(config.dropout)

        # 定义 mask
        self.register_buffer('mask', torch.tril(torch.ones((self.context_length, self.context_length))))

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
                B：batch size：批次大小
                T: Time steps(current context_length)：训练时当前context_length
                C: Channels(dimensions): d_model
        """
        # 验证X的形状
        B, T, C = x.shape
        assert T <= self.context_length
        assert C == self.d_model

        # 准备Q、K、V
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # 缩放点积注意力（Scaled dot product attention）: Q @ K^T / sqrt(d_k)
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 掩码覆盖
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # 计算softmax
        weights = F.softmax(input=weights, dim=-1)

        # 随机丢弃
        weights = self.dropout_layer(weights)

        # 缩放点积注意力（Scaled dot product attention）: weights @ V
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, config: FutureAiConfig):
        super().__init__()

        # 常规变量定义
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.d_model = config.d_model
        self.context_length = config.context_length

        # 多头注意力
        self.heads = nn.ModuleList([Attention(config=config) for _ in range(self.num_heads)])

        # 定义Wo
        self.Wo = nn.Linear(in_features=self.d_model, out_features=self.d_model)

        # 随机丢弃
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, x):
        # 多头注意力计算 & 合并
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # 应用Wo
        out = self.Wo(out)

        # 随机丢弃
        out = self.dropout_layer(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: FutureAiConfig):
        super().__init__()

        # 配置参数获取
        self.d_model = config.d_model
        self.context_length = config.context_length
        self.head_size = config.head_size
        self.num_heads = config.num_heads
        self.dropout = config.dropout

        # 多头注意力
        self.multi_head_attention_layer = MultiHeadAttention(config=config)

        # 前馈神经网络
        self.feed_forward_layer = FeedForward(config=config)

        # 两个归一化层
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        # Note: The order of the operations is different from the original Transformer paper
        # The order here is:
        """
        此处的处理方式与 Transformers 原论文有点不同，调整了两个 Layer_norm 的顺序
        LayerNorm -> Multi-head attention -> LayerNorm -> Feed forward
        """
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))
        x = x + self.feed_forward_layer(self.layer_norm_2(x))
        return x


class FutureAiModel(PreTrainedModel):
    def __init__(self, config: FutureAiConfig):
        super().__init__()

        # 配置参数获取
        self.d_model = config.d_model
        self.num_blocks = config.num_blocks
        self.num_heads =  config.num_heads
        self.context_length = config.context_length
        self.dropout = config.dropout
        self.max_token_value = config.max_token_value

        # 定义输入 token embedding 查找表格
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value, embedding_dim=self.d_model)

        # 定义 Transformer Blocks
        # 此处与原论文不太相同，在所有的 block 结束后添加了一个归一化层（LayerNorm）
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(config=config) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))

        # 定义输出线性层
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)

        self.register_buffer('position_encoding_lookup_table', self.init_position_encoding())

    def init_position_encoding(self):
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        return position_encoding_lookup_table

    def forward(self, input_ids, targets = None, **keyargs):
        idx = input_ids
        targets = input_ids if targets is None else targets
        # 获取输入形状
        B, T = idx.shape

        # 获取位置编码
        position_embedding = self.position_encoding_lookup_table[:T, :].to(idx.device)

        # 输入token向量化 + 位置编码
        print(f"idx:{idx},max_token_value:{self.max_token_value}")
        x = self.token_embedding_lookup_table(idx) + position_embedding

        # 循环执行 Transformer Blocks
        x = self.transformer_blocks(x)

        # 输出线性层处理
        logits = self.language_model_out_linear_layer(x)

        # 调试信息
        print(f"logits shape: {logits.shape}")
        print(f"targets shape: {targets.shape}" if targets is not None else "No targets provided")

        # 计算损失函数并执行返回
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
            print(f"Loss: {loss}")
            if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
                print("Loss is not a scalar")
            else:
                print("Loss is a scalar")
        else:
            loss = None
        # return logits, loss
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits
        )

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        """
        生成新的文本序列
        Args:
            idx: 形状为 (B, T)，其中 B 是批次大小，T 是时间步长
            max_new_tokens: 需要生成的新 token 数量
        """
        # 循环生成新的token
        for _ in range(max_new_tokens):
            # 截取 idx 的最后 context_length 个token
            idx_crop = idx[:, -self.context_length:]
            # 使用模型预测这些 token 的下一个 token 的 logits, loss
            logits, loss = self(idx_crop)
            # 获取最后一个时间步的 logits
            logits_last_timestep = logits[:, -1, :]
            # 应用 softmax 转换为概率分布
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # 根据概率分布采样得到下一个 token 的索引
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # 将新生成的 token 添加到 idx 中
            idx = torch.cat((idx, idx_next), dim=1)
        # 执行返回
        return idx
