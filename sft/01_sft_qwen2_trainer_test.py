from transformers import AutoTokenizer

from model_define.model_qwen.modeling_qwen2 import Qwen2ForCausalLM
import torch


"""
MODULE 0 定义任务类型
city：省会信息
else：省份介绍
"""
eval_prompt_list = ['解释一下深度学习中的过拟合问题','简述自然语言处理的主要任务','列举几个常用的机器学习框架']


"""
MODULE 1 定义运行设备
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'


"""
MODULE 2 Tokenizer 与 Model 定义
"""
model_type = "sft"
model_path = f"../checkpoint_folder/sft/qwen2-sft-qa/checkpoint-4000"
tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")
model = Qwen2ForCausalLM.from_pretrained(model_path)


"""
MODULE 3 生成测试
"""
for prompt in eval_prompt_list:
    for i in range(1):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_return_sequences=1,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id
        )

        response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print('---------------')
        print(response)
        print('---------------')