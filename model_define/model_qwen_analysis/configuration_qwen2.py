# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Qwen2 model configuration
Qwen2 模型配置类
"""
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    这是用于存储 [`Qwen2Model`] 配置的配置类。它用于根据指定的参数实例化 Qwen2 模型，定义模型架构。
    使用默认值实例化将产生与 Qwen2-7B-beta 类似的配置。

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    TODO 此处说的可用于控制模型输出是什么意思？


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]

        词表大小：
            Qwen2 模型的词汇量，定于调用 [`Qwen2Model`] 时传递的 `inputs_ids` 可以表示的不同标记的数量。

        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.

        隐藏层维度大小：
            可以理解为 Transformer 中每一层输入和输出维度的大小
            (1) 输入和输出维度一致：在Transformer模型中，每个编码器层和解码器层的输入和输出具有相同的维度，这个维度就是 hidden_size
            (2) 由于 input embedding 的结果作为第一层的输入，所以 input embedding 的维度也是 hidden_size
            (2) 前馈神经网络（FFN）中的维度：虽然在每个Transformer层内部，前馈神经网络（Feed-Forward Network, FFN）中间层的维度可以
            比 hidden_size 大（例如在很多实现中是 hidden_size * 4），但是FFN的输入和输出维度仍然保持为 hidden_size。这是为了确保层
            与层之间的兼容性，使得上一层的输出可以直接作为下一层的输入。
            (3) 自注意力机制中的维度：在自注意力机制中，查询（Q）、键（K）和值（V）矩阵的维度通常是 hidden_size 或者
            是 hidden_size / num_heads（对于多头注意力机制而言），其中 num_heads 是多头的数量。然而，经过多头注意力机制处理后的输出
            会再次映射回 hidden_size 维度，以便与FFN以及其他层进行连接。

        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.

        中间层维度：
            MLP 中间层维度
            MLP（Multilayer Perceptron，多层感知机）是每一层中的前馈神经网络部分（Feed-Forward Neural Network, FFN）

        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.

        隐藏层数量：
            Transformer Encoder 层数

        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.

        注意力头数：
            Transformer Encoder 每一层注意力模块的注意力头数

        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.

        kv头数
            分组查询注意力（GQA，Grouped Query Attention）的kv头数
            当num_key_value_heads=num_attention_heads时，将使用多头注意力 (MHA, Multi Head Attention)
            当num_key_value_heads=1时，将使用多查询注意力 (MQA, Multi Query Attention)
            其他情况，使用分组查询注意力（GQA，Grouped Query Attention）

            更多的信息可以查看：https://arxiv.org/pdf/2305.13245.pdf
            TODO 此处不是很懂不同注意力区别

        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.

        隐藏层行为：
            MLP中的激活函数

        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.

        最大位置嵌入：
            模型支持的最大序列长度

        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

        初始化范围：
            truncated_normal_initializer 用于初始化所有权重矩阵的标准差
            TODO 此处不是很明白

        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.

        归一化层 eps:
            均方根归一化层使用的 epsilon。用于防止除零错误，是一个非常小的正数

        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.

        是否使用缓存：
            模型是否应返回最后的键/值注意力（并非所有模型都使用）
            仅当 `config.is_decoder=True` 时才有效
            TODO 最后的键、值注意力是什么意思

        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

        绑定词向量：
            模型的输入和输出的词向量是否应该绑定
            TODO 不是很明白

        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.

        rope_theta：
            RoPE（Rotary Position Embeddings） 嵌入的基础周期
            TODO 不是很明白

        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.

            RoPE 配置参数

            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE

        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.

        是否使用滑动窗口注意力：
            是否使用滑动窗口注意力（SWA, Sliding window attention）
            TODO 什么时滑动窗口注意力

        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.

        滑动窗口注意力窗口大小

        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.

        滑动窗口注意力层数：
            底层使用SWA，顶层使用全注意力机制
            TODO 底层和顶层哪一个离输入近？全注意力又是什么

        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

        注意力层 dropout 比率


    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
