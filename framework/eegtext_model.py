import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class ProjectionHead(nn.Module):
    """
    将不同模态的 encoder feature
    投影到统一的 shared embedding space。
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        return self.projection(x)


class EEGTextModel(nn.Module):
    """
    EEG-Text multimodal alignment model.

    EEG:
        eeg
        -> eeg_encoder
        -> eeg_projection
        -> normalize
        -> z_eeg

    Text:
        text
        -> text_encoder
        -> text_projection
        -> normalize
        -> z_text
    """

    def __init__(
        self,
        eeg_encoder: nn.Module,
        text_encoder: nn.Module,

        eeg_feature_dim: int | None = None,
        text_feature_dim: int | None = None,

        projection_dim: int = 256,
        projection_hidden_dim: int = 512,

        freeze_eeg_encoder: bool = False,
        freeze_text_encoder: bool = True,

        temperature: float = 0.07,
    ):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
            
        eeg_feature_dim = self._feature_dim(eeg_encoder, eeg_feature_dim)
        text_feature_dim = self._feature_dim(text_encoder, text_feature_dim)

        # ====================================================
        # 1. Encoders
        # ====================================================

        self.eeg_encoder = eeg_encoder
        self.text_encoder = text_encoder

        # ====================================================
        # 2. Projection heads
        # ====================================================

        self.eeg_projection = ProjectionHead(
            input_dim=eeg_feature_dim,
            projection_dim=projection_dim,
            hidden_dim=projection_hidden_dim,
        )

        self.text_projection = ProjectionHead(
            input_dim=text_feature_dim,
            projection_dim=projection_dim,
            hidden_dim=projection_hidden_dim,
        )

        # ====================================================
        # 3. Temperature
        # ====================================================

        self.temperature = temperature

        # ====================================================
        # 4. Freeze encoders
        # ====================================================

        self.set_encoder_trainable("eeg", not freeze_eeg_encoder)
        self.set_encoder_trainable("text", not freeze_text_encoder)

    @staticmethod
    def _feature_dim(encoder, explicit_dim):
        inferred_dim = getattr(encoder, "feature_dim", None)
        if explicit_dim is not None and inferred_dim is not None and explicit_dim != inferred_dim:
            raise ValueError("Explicit feature dimension does not match encoder.feature_dim")
        dimension = explicit_dim if explicit_dim is not None else inferred_dim
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Provide a positive feature dimension or encoder.feature_dim")
        return dimension

    def set_encoder_trainable(self, modality: str, trainable: bool):
        """统一设置梯度及训练模式；投影头始终独立训练。"""
        if modality not in ("eeg", "text"):
            raise ValueError("modality must be 'eeg' or 'text'")
        encoder = getattr(self, f"{modality}_encoder")
        encoder.requires_grad_(trainable)
        encoder.train(self.training if trainable else False)

    def train(self, mode: bool = True):
        super().train(mode)
        for encoder in (self.eeg_encoder, self.text_encoder):
            if not any(param.requires_grad for param in encoder.parameters()):
                encoder.eval()
        return self

    # ========================================================
    # EEG branch
    # ========================================================

    def encode_eeg(self, eeg):
        """
        eeg:
            (B, C, T)

        return:
            eeg_feature: encoder 原始输出
            eeg_embedding: 投影 + normalize 后的输出
        """

        eeg_feature = self.eeg_encoder(eeg)

        # 如果 EEG encoder 输出还有额外维度：
        #
        # (B, D, T')
        #
        # 可以在这里做 pooling
        if eeg_feature.ndim == 3:
            eeg_feature = eeg_feature.mean(dim=-1)

        eeg_embedding = self.eeg_projection(eeg_feature)

        # 对它们做 L2 归一化后，点积就直接等价于余弦相似度。
        eeg_embedding = F.normalize(
            eeg_embedding,
            p=2,
            dim=-1
        )

        return eeg_feature, eeg_embedding

    # ========================================================
    # Text branch
    # ========================================================

    def encode_text(self, text):
        """
        假设 text_encoder 最终输出：

            (B, 768)

        return:
            text_feature: BERT feature
            text_embedding: 投影 + normalize 后的 feature
        """

        text_feature = self.text_encoder(text)

        text_embedding = self.text_projection(text_feature)

        text_embedding = F.normalize(
            text_embedding,
            p=2,
            dim=-1
        )

        return text_feature, text_embedding

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, eeg, text):

        eeg_feature, eeg_embedding = self.encode_eeg(eeg)

        text_feature, text_embedding = self.encode_text(text)

        return {
            "eeg_feature": eeg_feature,
            "text_feature": text_feature,

            "eeg_embedding": eeg_embedding,
            "text_embedding": text_embedding,
        }

    # ========================================================
    # Contrastive loss
    # ========================================================

    def contrastive_loss(
        self,
        eeg_embedding,
        text_embedding,
    ):
        """
        CLIP-style symmetric contrastive loss.

        eeg_embedding:
            (B, D)

        text_embedding:
            (B, D)
        """

        # --------------------------------------------
        # cosine similarity
        #
        # 因为前面已经 normalize：
        #
        # z1 @ z2.T == cosine similarity
        # --------------------------------------------

        if (
            eeg_embedding.ndim != 2
            or text_embedding.ndim != 2
            or eeg_embedding.shape != text_embedding.shape
            or eeg_embedding.shape[0] == 0
        ):
            raise ValueError("Embeddings must have the same non-empty (B, D) shape")
        logits = (
            eeg_embedding @ text_embedding.T
        ) / self.temperature

        batch_size = eeg_embedding.size(0)

        labels = torch.arange(
            batch_size,
            device=eeg_embedding.device
        )

        # EEG -> Text
        loss_eeg_to_text = F.cross_entropy(
            logits,
            labels
        )

        # Text -> EEG
        loss_text_to_eeg = F.cross_entropy(
            logits.T,
            labels
        )

        loss = (
            loss_eeg_to_text
            + loss_text_to_eeg
        ) / 2

        return loss

    # ========================================================
    # Complete training forward
    # ========================================================

    def compute_loss(self, eeg, text):

        outputs = self.forward(
            eeg=eeg,
            text=text,
        )

        loss = self.contrastive_loss(
            outputs["eeg_embedding"],
            outputs["text_embedding"],
        )

        outputs["loss"] = loss

        return outputs
