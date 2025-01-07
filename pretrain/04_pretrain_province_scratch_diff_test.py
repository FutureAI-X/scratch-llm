from transformers import AutoTokenizer

import torch

from model_define.model_futureai_diff.configuration_futureai import FutureAiConfig
from model_define.model_futureai_diff.modeling_futureai import FutureAiModelDiff


"""
MODULE 0 定义任务类型
city：省会信息
else：省份介绍
"""
dataset_type = "city"
model_save_name = "scratch-diff-province-city" if dataset_type == "city" else "scratch-diff-province-intro"
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
config = FutureAiConfig()
config.context_length = 32
config.d_model = 64
config.num_heads = 8
config.head_size = 8
config.num_blocks = 4

# 2.3 初始化模型
model = FutureAiModelDiff(config)

# 2.4 加载预训练参数
model.load_state_dict(torch.load(f'../checkpoint_folder/checkpoint_province/{model_save_name}.pt', weights_only=True))

# 2.5 将模型加载到设备上
model = model.to(device)


"""
MODULE 3 生成测试
"""
# 3.1 启用评估模式
model.eval()
for start in eval_prompt_list:
    start_ids = tokenizer.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    y = model.generate(x, max_new_tokens=32)
    print('---------------')
    print(tokenizer.decode(y[0].tolist()))
    print('---------------')