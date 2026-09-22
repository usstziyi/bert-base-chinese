from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModel.from_pretrained(
    "bert-base-chinese",
    output_hidden_states=True
)

model.eval()

text = "小王子住在一颗小行星上。"

inputs = tokenizer(
    text,
    return_tensors="pt",
    return_special_tokens_mask=True
)

with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

# outputs.hidden_states:
# 13 个元素
# embedding layer + 12 个 Transformer layer

last_four = torch.stack(
    outputs.hidden_states[-4:],
    dim=0
)
print(last_four.shape)

# (4, batch, seq_len, 768)

hidden = last_four.mean(dim=0)

# (batch, seq_len, 768)

mask = (
    inputs["attention_mask"]
    * (1 - inputs["special_tokens_mask"])
).unsqueeze(-1)

sentence_embedding = (
    (hidden * mask).sum(dim=1)
    / mask.sum(dim=1)
)

print(sentence_embedding.shape)