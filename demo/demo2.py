from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

text = "我喜欢自然语言处理"

"""
tokenizer,返回 input_ids, attention_mask, token_type_ids
"""

# text -> tokens -> input_ids, attention_mask, token_type_ids
inputs = tokenizer(
    text,
    add_special_tokens=True
)

print(inputs["input_ids"])
print(inputs["attention_mask"])
print(inputs["token_type_ids"])

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
print(tokens)