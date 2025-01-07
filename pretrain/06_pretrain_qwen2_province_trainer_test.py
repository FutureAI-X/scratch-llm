from transformers import AutoTokenizer

from model_define.model_qwen.modeling_qwen2 import Qwen2ForCausalLM
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = "../checkpoint_folder/checkpoint_province/qwen2-source/checkpoint-1650"
tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")
model = Qwen2ForCausalLM.from_pretrained(model_path)

for i in range(10):
    for start in ['杭州', '南京', '你好']:
        inputs = tokenizer(start, return_tensors="pt")

        generate_ids = model.generate(inputs.input_ids, max_new_tokens=10)
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print('---------------')
        print(response)
        print('---------------')