from transformers import AutoTokenizer

import torch

from model_define.model_qwen.configuration_qwen2 import Qwen2Config
from model_define.model_qwen.modeling_qwen2 import Qwen2ForCausalLM


"""
MODULE 0 定义任务类型
city：省会信息
else：省份介绍
"""
dataset_type = "city"
model_save_name = "qwen2-province-city" if dataset_type == "city" else "qwen2-province-intro"
eval_prompt_list = ['杭州','广州','成都','哈尔滨', '你好'] if dataset_type == "city" else ['浙江','广东','四川','黑龙江', '你好']


"""
MODULE 1 定义运行设备
"""
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

# 2.4 加载预训练参数
model.load_state_dict(torch.load(f'../checkpoint_folder/checkpoint_province/{model_save_name}.pt', weights_only=True))

# 2.5 将模型加载到设备上
model = model.to(device)

"""
MODULE 3 生成测试
"""
# 3.1 启用评估模式
model.eval()
for start in ['杭州','南京','你好']:
    for i in range(3):
        start_ids = tokenizer.encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        y = model.generate(x, max_new_tokens=32)
        print('---------------')
        print(tokenizer.decode(y[0].tolist()))
        print('---------------')