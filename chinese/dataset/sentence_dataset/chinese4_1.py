import os
import numpy as np
import torch
import openpyxl
from transformers import AutoTokenizer, AutoModel

# ============================================================
# 1. Configuration
# ============================================================

NOVEL_NAME = "LittlePrince"
RUN_INDEX = 1

# 和你生成 embedding 时的目录保持一致
EEG_ROOT = "D:/AI/ChineseEEG"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs"))
EMBEDDING_ROOT = os.path.join(OUTPUT_ROOT, "embeddings", "sentence_level", NOVEL_NAME)

# ============================================================
# 2. Device + tokenizer + BERT model
# 用于第 7 节：重新计算一句 embedding，和保存的结果做对比
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[Device] CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[Device] CUDA is NOT available. Using CPU.")

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese", output_hidden_states=True)
model = model.to(device)
model.eval()

# ============================================================
# 3. Load one run
# ============================================================

def load_embedding_run(embedding_root, run_index):

    embedding_path = os.path.join(embedding_root, f"text_embedding_run_{run_index}.npy")

    # Load NumPy file
    embeddings = np.load(embedding_path)

    # Basic shape check
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 768

    return embeddings

# ============================================================
# 4. Load original texts
# 用于：
#   1. 检查句子数量是否一致
#   2. 取出某一句原文，配合 embedding 查看
# ============================================================

def load_run_texts(novel_name, run_index):

    # 注意：
    # 数据文件名本身拼写为 "Chinense"
    # 而不是 "Chinese"
    segmented_path = os.path.join(
        EEG_ROOT, "derivatives", "novels", "segmented_novel", novel_name
    )
    novel_path = os.path.join(
        segmented_path, f"segmented_Chinense_novel_run_{run_index}.xlsx"
    )

    wb = openpyxl.load_workbook(novel_path, read_only=True, data_only=True)
    wsheet = wb.active

    texts = []
    for row in range(2, wsheet.max_row + 1):
        text = wsheet.cell(row=row, column=1).value
        if text is not None and str(text).strip():
            texts.append(str(text).strip())

    wb.close()
    return texts

# ============================================================
# 5. Load
# ============================================================

embeddings = load_embedding_run(EMBEDDING_ROOT, RUN_INDEX)
texts = load_run_texts(NOVEL_NAME, RUN_INDEX)

# ============================================================
# 6. Print shapes
# ============================================================

print("\n===== Dataset shapes =====")
print("embeddings:", embeddings.shape)
print("num texts: ", len(texts))

assert embeddings.shape[0] == len(texts)

# ============================================================
# 7. Inspect one sentence
# ============================================================

sentence_index = 1

print("\n===== Sentence inspection =====")
print("Sentence index:", sentence_index)
print("Text:")
print(texts[sentence_index])

sentence_embedding = embeddings[sentence_index]

print("\nEmbedding shape:", sentence_embedding.shape)
print("L2 norm:        ", np.linalg.norm(sentence_embedding))
print("Mean:           ", sentence_embedding.mean())
print("Std:            ", sentence_embedding.std())
print("First 8 dims:")
print(sentence_embedding[:8])

# ============================================================
# 8. Verify: recompute one sentence embedding
# 完全复现 chinese4.py 的流程：
#   1. 最后 4 层 Transformer 取平均
#   2. 排除 [CLS] [SEP] [PAD] 后做 mean pooling
# ============================================================

print("\n===== Verification (recompute) =====")

with torch.no_grad():
    inputs = tokenizer(
        texts[sentence_index],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    # Last 4 layers -> mean
    last_four = torch.stack(outputs.hidden_states[-4:], dim=0)
    hidden = last_four.mean(dim=0)

    # 排除特殊 token 后 mean pooling
    mask = inputs["attention_mask"] * (1 - inputs["special_tokens_mask"])
    mask = mask.unsqueeze(-1)

    recomputed = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

recomputed = recomputed.squeeze(0).cpu().numpy()

# 对比：余弦相似度 + 最大绝对误差
cos_sim = np.dot(embeddings[sentence_index], recomputed) / (
    np.linalg.norm(embeddings[sentence_index]) * np.linalg.norm(recomputed)
)
max_abs_diff = np.abs(embeddings[sentence_index] - recomputed).max()

print("Saved embedding vs recomputed:")
print("Cosine similarity:", cos_sim)
print("Max abs diff:     ", max_abs_diff)

if cos_sim > 0.999:
    print("OK: saved embedding matches recomputed embedding.")
else:
    print("WARNING: mismatch detected.")
