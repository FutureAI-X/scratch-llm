import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from model_define.model_futureai_diff.configuration_futureai import FutureAiConfig


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

def lambda_init_fn(depth):
    depth = depth + 1
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class AttentionDiff(nn.Module):
    def __init__(self, config: FutureAiConfig, depth: int):
        super().__init__()

        # 定义常规变量
        self.d_model = config.d_model
        self.head_size = config.head_size
        self.context_length = config.context_length

        # 定义Wq、Wk、Wv
        self.Wq = nn.Linear(in_features=self.d_model, out_features=2 * self.head_size, bias=False)
        self.Wk = nn.Linear(in_features=self.d_model, out_features=2 * self.head_size, bias=False)
        self.Wv = nn.Linear(in_features=self.d_model, out_features=2 * self.head_size, bias=False)

        # 定义 dropout
        self.dropout_layer = nn.Dropout(config.dropout)

        # 定义 mask
        self.register_buffer('mask', torch.tril(torch.ones((self.context_length, self.context_length))))

        # 以下为 Diff Transformers 增加的代码
        self.scale = self.head_size ** -0.5
        self.depth = depth

        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_size, dtype=torch.float32).normal_(mean=0, std=0.1))

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

        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        # 缩放点积注意力（Scaled dot product attention）: Q @ K^T / sqrt(d_k)
        weights1 = (q1 @ k1.transpose(-2, -1)) * (1.0 / math.sqrt(k1.size(-1)))
        weights2 = (q2 @ k2.transpose(-2, -1)) * (1.0 / math.sqrt(k2.size(-1)))

        # 掩码覆盖
        weights1 = weights1.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights2 = weights2.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q2)
        lambda_init = lambda_init_fn(self.depth)
        lambda_ = lambda_1 - lambda_2 + lambda_init

        # 计算softmax
        weights = (F.softmax(input=weights1, dim=-1) - lambda_ * F.softmax(input=weights2, dim=-1))

        # 随机丢弃
        weights = self.dropout_layer(weights)

        # 缩放点积注意力（Scaled dot product attention）: weights @ V
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, config: FutureAiConfig, depth: int):
        super().__init__()

        # 常规变量定义
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.d_model = config.d_model
        self.context_length = config.context_length

        # 多头注意力
        self.heads = nn.ModuleList([AttentionDiff(config=config, depth=depth) for _ in range(self.num_heads)])

        # 定义Wo
        self.Wo = nn.Linear(in_features=2 * self.d_model, out_features=self.d_model)

        # 随机丢弃
        self.dropout_layer = nn.Dropout(config.dropout)

        self.lambda_init = lambda_init_fn(depth)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.head_size * 2)

    def forward(self, x):
        # 多头注意力计算 & 合并
        out = torch.cat([self.layer_norm(h(x)) for h in self.heads], dim=-1)

        out = out * (1 - self.lambda_init)

        # 应用Wo
        out = self.Wo(out)

        # 随机丢弃
        out = self.dropout_layer(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: FutureAiConfig, depth: int):
        super().__init__()

        # 配置参数获取
        self.d_model = config.d_model
        self.context_length = config.context_length
        self.head_size = config.head_size
        self.num_heads = config.num_heads
        self.dropout = config.dropout

        # 多头注意力
        self.multi_head_attention_layer = MultiHeadAttention(config=config, depth=depth)

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


class FutureAiModelDiff(PreTrainedModel):

    config_class = FutureAiConfig

    def __init__(self, config: FutureAiConfig):
        super().__init__(config)

        # 配置参数获取
        self.d_model = config.d_model
        self.num_blocks = config.num_blocks
        self.num_heads =  config.num_heads
        self.context_length = config.context_length
        self.dropout = config.dropout
        self.vocab_size = config.vocab_size

        # 定义输入 token embedding 查找表格
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)

        # 定义 Transformer Blocks
        # 此处与原论文不太相同，在所有的 block 结束后添加了一个归一化层（LayerNorm）
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(config=config, depth=depth) for depth in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))

        # 定义输出线性层
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.vocab_size)

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
        x = self.token_embedding_lookup_table(idx) + position_embedding

        # 循环执行 Transformer Blocks
        x = self.transformer_blocks(x)

        # 输出线性层处理
        logits = self.language_model_out_linear_layer(x)

        # 计算损失函数并执行返回
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits
        )

    @torch.inference_mode()
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
            inference_res = self(idx_crop)
            # 获取最后一个时间步的 logits
            logits = inference_res.logits
            logits_last_timestep = logits[:, -1, :]
            # 应用 softmax 转换为概率分布
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # 根据概率分布采样得到下一个 token 的索引
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # 将新生成的 token 添加到 idx 中
            idx = torch.cat((idx, idx_next), dim=1)
        # 执行返回
        return idx
