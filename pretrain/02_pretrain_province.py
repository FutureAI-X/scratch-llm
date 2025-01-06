import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model_define.configuration_futureai import FutureAiConfig
from model_define.modeling_futureai import FutureAiModel

from tqdm import tqdm

learning_rate = 1e-3
epochs = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("../tokenizer_file/tokenizer_qwen")

# 定义Model
config = FutureAiConfig()
model = FutureAiModel(config)
model = model.to(device)

dataset = load_dataset('json', data_files="../dataset_file/dataset_province/province_city.jsonl", split='train')

def tokenize_function(batch):
    """对原始文本进行token化，生成新的参数：
        input_ids       token化之后的数值
        attention_mask  损失掩码，为1表示需要计算损失，为0表示不计算损失
    """
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=config.context_length + 1)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


def custom_collate_fn(batch):
    x = torch.stack([torch.tensor(item['input_ids'][:-1]) for item in batch]).to(device)
    y = torch.stack([torch.tensor(item['input_ids'][1:]) for item in batch]).to(device)
    return {
        'x': x,
        'y': y
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
        x = batch['x']
        y = batch['y']

        outputs = model(input_ids = x, targets=y)

        loss = outputs.loss

        print(f"Epoch {epoch}, Loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        i = i + 1

        print(i)

        if i % 100 == 0:
            torch.save(model.state_dict(), '../checkpoint_folder/checkpoint_province/model-province-scratch.pt')

torch.save(model.state_dict(), '../checkpoint_folder/checkpoint_province/model-province-scratch.pt')