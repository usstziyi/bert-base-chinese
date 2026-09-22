from transformers import AutoTokenizer, AutoModel
import openpyxl
import os
import numpy as np
import torch
import argparse


# ============================================================
# 1. Device + tokenizer + BERT model
# ============================================================

# ------------------------------------------------------------
# Device selection: prefer CUDA if available, else CPU
# ------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[Device] CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[Device] CUDA is NOT available. Using CPU.")

# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# ------------------------------------------------------------
# Load BERT model
# ------------------------------------------------------------
model = AutoModel.from_pretrained("bert-base-chinese", output_hidden_states=True)

# Move model to selected device
model = model.to(device)

# Evaluation mode: disable dropout, etc.
model.eval()


# ============================================================
# 2. Novel configuration (cross-platform paths)
# ============================================================

# 数据集根目录：
#
# 1. 优先读取环境变量 CHINESE_EEG_ROOT
# 2. Windows 默认：D:/AI/ChineseEEG
# 3. Linux/macOS 默认：~/ChineseEEG
if "CHINESE_EEG_ROOT" in os.environ:
    EEG_ROOT = os.environ["CHINESE_EEG_ROOT"]
elif os.name == "nt":
    EEG_ROOT = "D:/AI/ChineseEEG"
else:
    EEG_ROOT = os.path.expanduser("~/ChineseEEG")

# ------------------------------------------------------------
# Output root
# ------------------------------------------------------------
# 固定为脚本所在目录的 ../outputs
# 不依赖当前工作目录 cwd
OUTPUT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
)

# ------------------------------------------------------------
# Novel configuration
# ------------------------------------------------------------
Chinese_novels = {
    "LittlePrince": {
        "segmented_path": os.path.join(
            EEG_ROOT, "derivatives", "novels", "segmented_novel", "LittlePrince"
        ),
        "run_num": 7,
        "embedding_path": os.path.join(OUTPUT_ROOT, "embeddings", "sentence_level", "LittlePrince"),
    },
    "GarnettDream": {
        "segmented_path": os.path.join(
            EEG_ROOT, "derivatives", "novels", "segmented_novel", "GarnettDream"
        ),
        "run_num": 18,
        "embedding_path": os.path.join(OUTPUT_ROOT, "embeddings", "sentence_level", "GarnettDream"),
    }
}


# ============================================================
# 3. Parse arguments
# ============================================================
parser = argparse.ArgumentParser(description="Extract BERT sentence embeddings")

parser.add_argument(
    "--novel_name",
    type=str,
    default="LittlePrince",
    choices=list(Chinese_novels.keys()),
    help="Novel name"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for BERT inference"
)

args = parser.parse_args()


# ============================================================
# 4. Load novel configuration
# ============================================================
novel_cfg = Chinese_novels[args.novel_name]

args.segmented_path = novel_cfg["segmented_path"]
args.run_num = novel_cfg["run_num"]
args.embedding_path = novel_cfg["embedding_path"]

os.makedirs(args.embedding_path, exist_ok=True)

print(
    f"[Config] Novel: {args.novel_name}\n"
    f"[Config] Batch size: {args.batch_size}\n"
    f"[Config] Input path: {args.segmented_path}\n"
    f"[Config] Output path: {args.embedding_path}"
)


# ============================================================
# 5. Extract embeddings for each run
# ============================================================
for i in range(args.run_num):

    # --------------------------------------------------------
    # Novel file path
    # --------------------------------------------------------
    # 注意：
    # 数据文件名本身拼写为 "Chinense"
    # 而不是 "Chinese"
    novel_path = os.path.join(
        args.segmented_path,
        f"segmented_Chinense_novel_run_{i + 1}.xlsx"
    )

    # print(f"\n[Run {i + 1}/{args.run_num}] Loading: {novel_path}")

    # ========================================================
    # Read Excel
    # ========================================================
    wb = openpyxl.load_workbook(novel_path, read_only=True, data_only=True)
    wsheet = wb.active

    texts = []
    for j in range(2, wsheet.max_row + 1):
        text = wsheet.cell(row=j, column=1).value

        # ----------------------------------------------------
        # Skip: None / "" / "   "
        # ----------------------------------------------------
        if text is not None and str(text).strip():
            texts.append(str(text).strip())

    wb.close()

    # print(f"[Run {i + 1}] Loaded {len(texts)} texts")

    # ========================================================
    # Extract sentence embeddings
    # ========================================================
    embeddings = []

    # --------------------------------------------------------
    # Batch processing
    # --------------------------------------------------------
    for start in range(0, len(texts), args.batch_size):
        end = min(start + args.batch_size, len(texts))
        batch_texts = texts[start:end]

        # ====================================================
        # Tokenization
        # ====================================================
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            # ----------------------------------------------
            # Dynamic padding:
            # 当前 batch 内所有句子
            # padding 到最长句子的长度
            # ----------------------------------------------
            padding=True,
            # ----------------------------------------------
            # BERT maximum sequence length = 512 tokens
            # ----------------------------------------------
            truncation=True,
            max_length=512,
            # 用于排除 [CLS] [SEP] [PAD]
            return_special_tokens_mask=True
        )

        # ----------------------------------------------------
        # Move all tensors to GPU / CPU
        # ----------------------------------------------------
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # ====================================================
        # BERT forward
        # ====================================================
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            # ------------------------------------------------
            # outputs.hidden_states
            #
            # length = 13
            #
            # hidden_states[0]  embedding layer
            # hidden_states[1]  Transformer Layer 1
            # ...
            # hidden_states[12] Transformer Layer 12
            # ------------------------------------------------

            # =================================================
            # Last 4 Transformer layers
            # =================================================
            # shape: (4, batch_size, seq_len, 768)
            last_four = torch.stack(outputs.hidden_states[-4:], dim=0)

            # =================================================
            # Average last 4 layers
            # =================================================
            # shape: (batch_size, seq_len, 768)
            hidden = last_four.mean(dim=0)

            # =================================================
            # Build token mask
            # =================================================
            # Remove: [CLS] ×  [SEP] ×  [PAD] ×
            # Keep:   normal token ✓
            mask = inputs["attention_mask"] * (1 - inputs["special_tokens_mask"])

            # shape: (batch_size, seq_len)
            mask = mask.unsqueeze(-1)

            # shape: (batch_size, seq_len, 1)

            # =================================================
            # Mean pooling
            # =================================================
            sentence_embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # shape: (batch_size, 768)

        # ====================================================
        # GPU -> CPU -> NumPy
        # ====================================================
        sentence_embeddings = sentence_embeddings.cpu().numpy()

        embeddings.append(sentence_embeddings)

        # print(f"[Run {i + 1}] Processed {end}/{len(texts)} texts")

    # ========================================================
    # Merge all batches
    # ========================================================
    # shape: (num_texts, 768)
    embeddings = np.concatenate(embeddings, axis=0)

    # ========================================================
    # Save embeddings
    # ========================================================
    save_path = os.path.join(args.embedding_path, f"text_embedding_run_{i + 1}.npy")

    np.save(save_path, embeddings)

    print(
        f"[Run {i + 1}] "
        f"{len(texts)} texts -> "
        f"{embeddings.shape}, "
        f"saved to:{save_path}"
    )


print("\nAll runs completed.")
