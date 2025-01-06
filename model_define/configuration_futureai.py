from transformers import PretrainedConfig

class FutureAiConfig(PretrainedConfig):
    """
    模型配置类
    Args:
        d_model: 输入 embedding 维度
        num_blocks: Transformer Block 数量
        num_heads: 注意力头数
        dropout: 随机丢弃比例，用于防止过度拟合。训练时可设置为0.1，推理时应设置为0.0
        context_length: 最大序列长度
        max_token_value：词表大小
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        head_size: int = 64,
        num_blocks: int = 8,
        dropout: float = 0.0,
        context_length: int = 128,
        vocab_size: int = 151644,
        **kwargs
    ):

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.context_length = context_length

        self.vocab_size = vocab_size

        super().__init__(**kwargs)