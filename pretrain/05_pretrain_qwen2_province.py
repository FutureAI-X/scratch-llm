import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model_define.model_qwen.configuration_qwen2 import Qwen2Config
from model_define.model_qwen.modeling_qwen2 import Qwen2ForCausalLM

learning_rate = 1e-3
epochs = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")

# 定义Model
config = Qwen2Config()
config.max_position_embeddings = 32

config.hidden_size = 64
config.num_attention_heads = 8
config.num_key_value_heads = 2
config.num_hidden_layers = 4
model = Qwen2ForCausalLM(config)
model = model.to(device)

dataset = load_dataset('json', data_files="../dataset_file/dataset_province/province_city.jsonl", split='train')

def tokenize_function(batch):
    """对原始文本进行token化，生成新的参数：
        input_ids       token化之后的数值
        attention_mask  损失掩码，为1表示需要计算损失，为0表示不计算损失
    """
    tokenizer_temp = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=config.max_position_embeddings + 1)
    tokenizer_temp["labels"] = tokenizer_temp["input_ids"].copy()
    return tokenizer_temp

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

def custom_collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch]).to(device)
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch]).to(device)
    labels = torch.stack([torch.tensor(item['labels']) for item in batch]).to(device)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

train_dataloader = DataLoader(
    tokenized_datasets,
    batch_size=2,
    shuffle=False,
    collate_fn=custom_collate_fn
)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    i = 0
    print(len(train_dataloader))
    for batch in train_dataloader:
        outputs = model(**batch)

        loss = outputs.loss

        print(f"Epoch {epoch}, Loss: {loss}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        i = i + 1

        print(i)

        if i % 100 == 0:
            torch.save(model.state_dict(), '../checkpoint_folder/checkpoint_province/model-qwen2-province-city.pt')
    torch.save(model.state_dict(), '../checkpoint_folder/checkpoint_province/model-qwen2-province-city.pt')

torch.save(model.state_dict(), '../checkpoint_folder/checkpoint_province/model-qwen2-province-city.pt')
