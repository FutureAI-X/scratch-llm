from transformers import AutoTokenizer

import torch

from model_define.model_futureai_diff.configuration_futureai import FutureAiConfig
from model_define.model_futureai_diff.modeling_futureai import FutureAiModelDiff

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")

config = FutureAiConfig()
config.context_length = 32
config.d_model = 64
config.num_heads = 8
config.head_size = 8
config.num_blocks = 4
model = FutureAiModelDiff(config)

model.load_state_dict(torch.load('../checkpoint_folder/checkpoint_province/model-province-city-diff.pt', weights_only=True))
model = model.to(device)

model.eval()
for start in ['杭州','南京','你好']:
    start_ids = tokenizer.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    y = model.generate(x, max_new_tokens=32)
    print('---------------')
    print(tokenizer.decode(y[0].tolist()))
    print('---------------')