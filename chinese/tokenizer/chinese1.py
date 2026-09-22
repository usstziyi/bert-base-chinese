from pathlib import Path

import numpy as np

abs_path = Path(r"D:/AI/ChineseEEG/derivatives/text_embeddings/LittlePrince_text_embedding")

x = np.load(abs_path / "text_embedding_run_1.npy")

print(x.shape)