import torch
from transformers import AutoTokenizer

from model_define.model_futureai.modeling_futureai import FutureAiModel

# 定义运行设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("./checkpoint_folder/checkpoint-59000")
model = FutureAiModel.from_pretrained("./checkpoint_folder/checkpoint-59000")

prompt = "她说"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_new_tokens=10)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(response)
