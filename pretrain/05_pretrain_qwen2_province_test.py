from transformers import AutoTokenizer

from model_define.model_qwen.configuration_qwen2 import Qwen2Config
from model_define.model_qwen.modeling_qwen2 import Qwen2ForCausalLM
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")

config = Qwen2Config()
config.max_position_embeddings = 32

config.hidden_size = 64
config.num_attention_heads = 8
config.num_key_value_heads = 2
config.num_hidden_layers = 4
model = Qwen2ForCausalLM(config)

model.load_state_dict(torch.load('../checkpoint_folder/checkpoint_province/model-qwen2-province-city.pt', weights_only=True))
model = model.to(device)

model.eval()
for start in ['杭州','南京','你好']:
    start_ids = tokenizer.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    y = model.generate(x, max_new_tokens=32)
    print('---------------')
    print(tokenizer.decode(y[0].tolist()))
    print('---------------')