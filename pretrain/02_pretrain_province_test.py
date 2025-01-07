from transformers import AutoTokenizer

from model_define.model_futureai.configuration_futureai import FutureAiConfig
from model_define.model_futureai.modeling_futureai import FutureAiModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")

config = FutureAiConfig()
config.context_length = 32
config.d_model = 64
config.num_heads = 8
config.head_size = 8
config.num_blocks = 4
model = FutureAiModel(config)

model.load_state_dict(torch.load('../checkpoint_folder/checkpoint_province/model-province-intro.pt', weights_only=True))
model = model.to(device)

model.eval()
for start in ['浙江','江苏','重庆']:
    start_ids = tokenizer.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    y = model.generate(x, max_new_tokens=32)
    print('---------------')
    print(tokenizer.decode(y[0].tolist()))
    print('---------------')