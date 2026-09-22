import os
import numpy as np
import torch
from transformers import AutoTokenizer

# ============================================================
# 1. Configuration
# ============================================================

NOVEL_NAME = "LittlePrince"
RUN_INDEX = 1

# 和你生成 embedding 时的目录保持一致
EEG_ROOT = "D:/AI/ChineseEEG"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs"))
EMBEDDING_ROOT = os.path.join(OUTPUT_ROOT, "embeddings", "word_level", NOVEL_NAME)

# ============================================================
# 2. Load tokenizer
# 主要用于：
# input_ids -> token
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# ============================================================
# 3. Load one run
# ============================================================

def load_embedding_run(embedding_root, run_index):

    embedding_path = os.path.join(embedding_root, f"text_embedding_run_{run_index}.npy")
    input_ids_path = os.path.join(embedding_root, f"input_ids_run_{run_index}.npy")
    attention_mask_path = os.path.join(embedding_root, f"attention_mask_run_{run_index}.npy")
    special_mask_path = os.path.join(embedding_root, f"special_tokens_mask_run_{run_index}.npy")

    # Load NumPy files
    embeddings = np.load(embedding_path)
    input_ids = np.load(input_ids_path)
    attention_masks = np.load(attention_mask_path)
    special_tokens_masks = np.load(special_mask_path)

    # Basic shape check
    num_sentences = embeddings.shape[0]
    assert input_ids.shape[0] == num_sentences
    assert attention_masks.shape[0] == num_sentences
    assert special_tokens_masks.shape[0] == num_sentences

    assert embeddings.shape[1] == input_ids.shape[1]
    assert input_ids.shape == attention_masks.shape
    assert input_ids.shape == special_tokens_masks.shape

    return {
        "embeddings": embeddings,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "special_tokens_mask": special_tokens_masks
    }

# ============================================================
# 4. Load
# ============================================================

data = load_embedding_run(EMBEDDING_ROOT, RUN_INDEX)

embeddings = data["embeddings"]
input_ids = data["input_ids"]
attention_mask = data["attention_mask"]
special_tokens_mask = data["special_tokens_mask"]

# ============================================================
# 5. Print shapes
# ============================================================

print("\n===== Dataset shapes =====")
print("embeddings:", embeddings.shape)
print("input_ids:", input_ids.shape)
print("attention_mask:", attention_mask.shape)
print("special_tokens_mask:", special_tokens_mask.shape)

# ============================================================
# 6. Inspect one sentence
# ============================================================

sentence_index = 1

sentence_input_ids = input_ids[sentence_index]
sentence_embeddings = embeddings[sentence_index]
sentence_attention_mask = attention_mask[sentence_index]
sentence_special_mask = special_tokens_mask[sentence_index]

# input_ids -> tokens
tokens = tokenizer.convert_ids_to_tokens(sentence_input_ids.tolist())

print("\n===== Sentence inspection =====")
print("Sentence index:", sentence_index)
print("Tokens:")
print(tokens)

print("\nInput IDs:")
print(sentence_input_ids)

print("\nAttention mask:")
print(sentence_attention_mask)

print("\nSpecial tokens mask:")
print(sentence_special_mask)

print("\nSentence embedding shape:")
print(sentence_embeddings.shape)

# ============================================================
# 7. Extract valid normal tokens
# special_tokens_mask:
# normal token  = 0
# special token = 1
# ============================================================

normal_token_mask = (sentence_special_mask == 0)
valid_tokens = [token for token, keep in zip(tokens, normal_token_mask) if keep]
valid_embeddings = sentence_embeddings[normal_token_mask]

print("\n===== Normal tokens only =====")
print(valid_tokens)
print("Valid embeddings shape:", valid_embeddings.shape)
