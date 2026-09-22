from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

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

hidden = outputs.last_hidden_state
# (1, seq_len, 768)


# 1 = 正文 token
# 0 = CLS / SEP / padding
mask = (
    inputs["attention_mask"]
    * (1 - inputs["special_tokens_mask"])
)


mask = mask.unsqueeze(-1)
# (1, seq_len, 1)

sentence_embedding = (
    (hidden * mask).sum(dim=1)
    / mask.sum(dim=1)
)

print(sentence_embedding.shape)