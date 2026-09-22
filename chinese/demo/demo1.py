from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

text = "我喜欢自然语言处理，和深度学习。"

# ① text -> tokens
tokens = tokenizer.tokenize(text)
print("tokens:")
print(tokens)

# ② tokens -> ids
ids = tokenizer.convert_tokens_to_ids(tokens)
print("\nids:")
print(ids)

"""
tokenizer.encode,只返回 input_ids
不返回 attention_mask, token_type_ids
"""

# ③ 正式 encode，加入特殊 token
encoded_ids = tokenizer.encode(
    text,
    add_special_tokens=True
)
print("\nencoded ids:")
print(encoded_ids)

# ④ id -> token
encoded_tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
print("\nencoded tokens:")
print(encoded_tokens)

# ⑤ 看 tokenizer 定义了哪些特殊 token
print("\nspecial tokens:")
print(tokenizer.special_tokens_map)