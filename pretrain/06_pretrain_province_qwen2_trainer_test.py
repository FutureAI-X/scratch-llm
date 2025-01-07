from transformers import AutoTokenizer

from model_define.model_qwen.modeling_qwen2 import Qwen2ForCausalLM
import torch


"""
MODULE 0 定义任务类型
city：省会信息
else：省份介绍
"""
dataset_type = "city"
model_save_name = "qwen2-province-city" if dataset_type == "city" else "qwen2-province-intro"
eval_prompt_list = ['杭州','哈尔滨','海口','乌鲁木齐'] if dataset_type == "city" else ['浙江', '黑龙江', '海南', '新疆']


"""
MODULE 1 定义运行设备
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'


"""
MODULE 2 Tokenizer 与 Model 定义
"""
model_path = f"../checkpoint_folder/checkpoint_province/{model_save_name}/checkpoint-2000"
tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")
model = Qwen2ForCausalLM.from_pretrained(model_path)


"""
MODULE 3 生成测试
"""
for start in eval_prompt_list:
    for i in range(3):
        inputs = tokenizer(start, return_tensors="pt")

        generate_ids = model.generate(inputs.input_ids, max_new_tokens=10)
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print('---------------')
        print(response)
        print('---------------')