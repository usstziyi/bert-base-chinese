from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os

NOVEL_NAME = "LittlePrince"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs"))
EMBEDDING_ROOT = os.path.join(OUTPUT_ROOT, "embeddings", "sentence_level", NOVEL_NAME)


class TextEmbeddingDataset(Dataset):

    def __init__(self, embedding_path):
        self.embeddings = torch.from_numpy(np.load(embedding_path)).float()

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, index):
        return {
            "embedding": self.embeddings[index]
        }




dataset = TextEmbeddingDataset(
    embedding_path=os.path.join(EMBEDDING_ROOT, "text_embedding_run_1.npy")
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


for batch in loader:
    print(batch["embedding"].shape)

    break
