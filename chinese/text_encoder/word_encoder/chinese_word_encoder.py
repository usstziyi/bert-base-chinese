from transformers import AutoTokenizer, AutoModel
import openpyxl
import os
import numpy as np
import torch
import argparse

# ============================================================
# 1. Device + tokenizer + BERT
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
# 2. Dataset configuration
# ============================================================

if "CHINESE_EEG_ROOT" in os.environ:
    EEG_ROOT = os.environ["CHINESE_EEG_ROOT"]
elif os.name == "nt":
    EEG_ROOT = "D:/AI/ChineseEEG"
else:
    EEG_ROOT = os.path.expanduser("~/ChineseEEG")

OUTPUT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../", "outputs")
)

Chinese_novels = {
    "LittlePrince": {
        "segmented_path": os.path.join(EEG_ROOT, "derivatives", "novels", "segmented_novel", "LittlePrince"),
        "run_num": 7,
        "embedding_path": os.path.join(OUTPUT_ROOT, "text_encoder_output", "word_level", "LittlePrince"),
    },
    "GarnettDream": {
        "segmented_path": os.path.join(EEG_ROOT, "derivatives", "novels", "segmented_novel", "GarnettDream"),
        "run_num": 18,
        "embedding_path": os.path.join(OUTPUT_ROOT, "text_encoder_output", "word_level", "GarnettDream"),
    }
}

# ============================================================
# 3. Arguments
# ============================================================

parser = argparse.ArgumentParser(description="Extract BERT token-level embeddings")
parser.add_argument("--novel_name", type=str, default="LittlePrince", choices=list(Chinese_novels.keys()))
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--max_length",
    type=int,
    default=None,
    help=(
        "Fixed token sequence length. "
        "If None, automatically use the longest sentence "
        "in the selected novel, capped at BERT's 512 tokens."
    )
)
args = parser.parse_args()

# ============================================================
# 4. Novel config
# ============================================================

novel_cfg = Chinese_novels[args.novel_name]
args.segmented_path = novel_cfg["segmented_path"]
args.run_num = novel_cfg["run_num"]
args.embedding_path = novel_cfg["embedding_path"]
os.makedirs(args.embedding_path, exist_ok=True)

# ============================================================
# 5. Helper: read one run
# ============================================================

def load_run_texts(run_index):
    novel_path = os.path.join(args.segmented_path, f"segmented_Chinense_novel_run_{run_index}.xlsx")
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
# 6. Load all runs
# 用于：
#   1. 避免重复读取 Excel
#   2. 计算整个小说统一的 max_length
# ============================================================

all_run_texts = []
for run_index in range(1, args.run_num + 1):
    texts = load_run_texts(run_index)
    all_run_texts.append(texts)
    print(f"[Run {run_index}] Loaded {len(texts)} texts")

# ============================================================
# 7. Determine global max_length
# ============================================================

# if args.max_length is None:
#     max_token_length = 0
#     for texts in all_run_texts:
#         for text in texts:
#             token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
#             max_token_length = max(max_token_length, len(token_ids))
#     # bert-base-chinese maximum = 512
#     max_length = min(max_token_length, 512)
#     print(f"[Tokenizer] Longest sequence: {max_token_length} tokens")
#     print(f"[Tokenizer] Using max_length = {max_length}")
# else:
#     if args.max_length > 512:
#         raise ValueError("bert-base-chinese supports at most 512 tokens.")
#     max_length = args.max_length
#     print(f"[Tokenizer] Using fixed max_length = {max_length}")

# ============================================================
# 8. Extract token-level embeddings
# ============================================================

for run_index, texts in enumerate(all_run_texts, start=1):

    # 每个 batch 的输出
    embedding_batches = []
    input_id_batches = []
    attention_mask_batches = []
    special_mask_batches = []

    # Batch processing
    for start in range(0, len(texts), args.batch_size):
        end = min(start + args.batch_size, len(texts))
        batch_texts = texts[start:end]

        # ====================================================
        # Tokenization
        # 关键变化：
        # padding="max_length"
        # 所有 batch 都统一成相同 seq_len
        # ====================================================

        max_length = 25

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )

        # 保存这些信息（注意这里先复制到 CPU）
        batch_input_ids = inputs["input_ids"].cpu().numpy()
        batch_attention_mask = inputs["attention_mask"].cpu().numpy()
        batch_special_mask = inputs["special_tokens_mask"].cpu().numpy()

        # Move model inputs to GPU
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        special_tokens_mask = inputs["special_tokens_mask"].to(device) # 用在BERT输出以后

        # ====================================================
        # BERT forward
        # ====================================================

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask  # attention_mask用于BERT内部的注意力机制
            )

            # Last four Transformer layers: 4 × batch × seq_len × 768
            last_four = torch.stack(outputs.hidden_states[-4:], dim=0)

            # =================================================
            # Average across the 4 layers
            # 注意：
            # 这里平均的是 Layer 维度，不是 Token 维度
            # shape: (batch, seq_len, 768)
            # =================================================

            token_embeddings = last_four.mean(dim=0)

            # =================================================
            # Zero PAD embeddings
            # attention_mask: normal token = 1, PAD = 0
            # shape: (batch, seq_len, 1)
            # =================================================

            token_mask = attention_mask.unsqueeze(-1)
            token_embeddings = token_embeddings * token_mask # 仅清零 PAD 位置的嵌入

            # token_mask = (1 - special_tokens_mask).unsqueeze(-1)
            # token_embeddings = token_embeddings * token_mask # 清零所有特殊 token 位置的嵌入

        # GPU -> CPU -> NumPy
        token_embeddings = token_embeddings.cpu().numpy()

        # Save this batch
        embedding_batches.append(token_embeddings)
        input_id_batches.append(batch_input_ids)
        attention_mask_batches.append(batch_attention_mask)
        special_mask_batches.append(batch_special_mask)


    # Merge batches
    embeddings = np.concatenate(embedding_batches, axis=0)
    input_ids = np.concatenate(input_id_batches, axis=0)
    attention_masks = np.concatenate(attention_mask_batches, axis=0)
    special_tokens_masks = np.concatenate(special_mask_batches, axis=0)

    # ========================================================
    # Shapes
    # embeddings:           (num_sentences, max_length, 768)
    # input_ids:            (num_sentences, max_length)
    # attention_masks:      (num_sentences, max_length)
    # special_tokens_masks: (num_sentences, max_length)
    # ========================================================

    # 9. Save
    embedding_path = os.path.join(args.embedding_path, f"text_embedding_run_{run_index}.npy")
    input_ids_path = os.path.join(args.embedding_path, f"input_ids_run_{run_index}.npy")
    attention_mask_path = os.path.join(args.embedding_path, f"attention_mask_run_{run_index}.npy")
    special_mask_path = os.path.join(args.embedding_path, f"special_tokens_mask_run_{run_index}.npy")

    np.save(embedding_path, embeddings)
    np.save(input_ids_path, input_ids)
    np.save(attention_mask_path, attention_masks)
    np.save(special_mask_path, special_tokens_masks)

    print(
        f"\n"
        f"[Run {run_index}]\n"
        f"Embeddings:         {embeddings.shape}\n"
        f"Input IDs:          {input_ids.shape}\n"
        f"Attention masks:    {attention_masks.shape}\n"
        f"Special masks:      {special_tokens_masks.shape}\n"
    )

print("\nAll runs completed.")


# uv run  .\chinese_word_encoder.py --novel_name GarnettDream
# uv run  .\chinese_word_encoder.py --novel_name LittlePrince