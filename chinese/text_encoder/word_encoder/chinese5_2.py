from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os

NOVEL_NAME = "LittlePrince"
EEG_ROOT = "D:/AI/ChineseEEG"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs"))
EMBEDDING_ROOT = os.path.join(OUTPUT_ROOT, "embeddings", "word_level", NOVEL_NAME)


class TextEmbeddingDataset(Dataset):

    def __init__(self, embedding_path, input_ids_path, attention_mask_path, special_mask_path):
        self.embeddings = torch.from_numpy(np.load(embedding_path)).float()
        self.input_ids = torch.from_numpy(np.load(input_ids_path)).long()
        self.attention_mask = torch.from_numpy(np.load(attention_mask_path)).long()
        self.special_tokens_mask = torch.from_numpy(np.load(special_mask_path)).long()

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, index):
        return {
            "embedding": self.embeddings[index],
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "special_tokens_mask": self.special_tokens_mask[index]
        }




dataset = TextEmbeddingDataset(
    embedding_path=os.path.join(EMBEDDING_ROOT, "text_embedding_run_1.npy"),
    input_ids_path=os.path.join(EMBEDDING_ROOT, "input_ids_run_1.npy"),
    attention_mask_path=os.path.join(EMBEDDING_ROOT, "attention_mask_run_1.npy"),
    special_mask_path=os.path.join(EMBEDDING_ROOT, "special_tokens_mask_run_1.npy")
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


for batch in loader:
    print(batch["embedding"].shape)
    print(batch["attention_mask"].shape)

    break