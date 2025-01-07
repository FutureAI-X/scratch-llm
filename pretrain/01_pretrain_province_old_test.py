from transformers import AutoTokenizer

from model_define.model_futureai.configuration_futureai import FutureAiConfig
from model_define.model_futureai.modeling_futureai import FutureAiModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")

config = FutureAiConfig()
model = FutureAiModel(config)

model.load_state_dict(torch.load('../checkpoint_folder/checkpoint_province/model-province.pt', weights_only=True))
model = model.to(device)

model.eval()
for i in range(10):
    start = '江苏'
    start_ids = tokenizer.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    y = model.generate(x, max_new_tokens=5)
    print('---------------')
    print(tokenizer.decode(y[0].tolist()))
    print('---------------')