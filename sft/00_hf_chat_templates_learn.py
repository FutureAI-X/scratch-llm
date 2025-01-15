from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")

messages = [{"content":"解释一下深度学习中的过拟合问题","role":"user"},{"content":"过拟合是模型在训练数据上表现很好但在新数据上表现差的现象主要是因为模型太复杂或训练数据不足","role":"assistant"}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

print(text)