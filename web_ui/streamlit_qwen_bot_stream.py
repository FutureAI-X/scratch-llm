from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import streamlit as st

model_name = "../checkpoint_folder/qwen_source/qwen2.5-0.5b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Use Thread to run generation in background
    # Otherwise, the process is blocked until generation is complete
    # and no streaming effect can be observed.
    from threading import Thread
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    return streamer

# 以上代码为 Qwen 官方提供的使用 Hugging Face Transformers 进行推理示例【流式输出】

# 以下代码为使用 Streamlit 构建聊天机器人，【流式输出】

st.title("🤖 Qwen ChatBot")

# 创建一个会话状态变量，用于存储聊天消息
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Input Something...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(user_input))
    st.session_state.messages.append({"role": "assistant", "content": response})