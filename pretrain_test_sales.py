from model_define.configuration_futureai import FutureAiConfig
from model_define.modeling_futureai import FutureAiModel
import torch
import tiktoken

device = 'cuda' if torch.cuda.is_available() else 'cpu'

encoding = tiktoken.get_encoding("cl100k_base")

config = FutureAiConfig()
config.vocab_size = 100070
config.d_model = 64
config.num_heads = 4
config.head_size = 16
config.context_length = 16
model = FutureAiModel(config)

model.load_state_dict(torch.load('./checkpoint_folder/checkpoint_sales/model-ckpt.pt'))
model = model.to(device)

model.eval()
start = 'The salesperson'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')