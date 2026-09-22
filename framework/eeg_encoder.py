from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class ResidualAdd(nn.Module):
    """Residual wrapper: y = x + fn(x)."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)


class TSConvEncoder(nn.Module):
    """
    TSConv feature extractor from:
    "Decoding Natural Images from EEG for Object Recognition"

    SA / GA are intentionally NOT included.

    Architecture:
        EEG
        -> Temporal Conv
        -> Avg Pool
        -> BatchNorm + ELU
        -> Spatial Conv
        -> BatchNorm + ELU
        -> Dropout
        -> 1x1 Conv
        -> Flatten

    Input:
        eeg: (B, C, T)

    Output:
        features: (B, D)

    Notes
    -----
    This implementation uses a fixed input_samples because the original
    TSConv encoder ends with Flatten + Linear.

    With the current dataset.py default:
        T = 1500
        k = 40
        temporal_kernel = 25
        pool_kernel = 51
        pool_stride = 5

    the temporal size becomes:
        1500 -> 1476 -> 286

    so the flattened dimension is:
        40 * 286 = 11440
    """

    def __init__(
        self,
        eeg_channels: int,
        input_samples: int = 1500,
        num_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 51,
        pool_stride: int = 5,
        emb_size: int = 40,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if eeg_channels <= 0:
            raise ValueError("eeg_channels must be > 0")
        if input_samples <= 0:
            raise ValueError("input_samples must be > 0")

        self.eeg_channels = eeg_channels
        self.input_samples = input_samples
        self.num_filters = num_filters
        self.temporal_kernel = temporal_kernel
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.emb_size = emb_size

        # ------------------------------------------------------------
        # 1. Temporal convolution
        #
        # (B, 1, C, T)
        # -> (B, k, C, T - m1 + 1)
        # ------------------------------------------------------------
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(1, temporal_kernel),
            stride=(1, 1),
            padding=0,
            bias=True,
        )

        # ------------------------------------------------------------
        # 2. Average pooling along the temporal dimension
        #
        # (B, k, C, T1)
        # -> (B, k, C, T2)
        # ------------------------------------------------------------
        self.avg_pool = nn.AvgPool2d(
            kernel_size=(1, pool_kernel),
            stride=(1, pool_stride),
        )

        self.bn1 = nn.BatchNorm2d(num_filters)
        self.elu1 = nn.ELU()

        # ------------------------------------------------------------
        # 3. Spatial convolution
        #
        # Kernel covers all EEG channels:
        # (B, k, C, T2)
        # -> (B, k, 1, T2)
        # ------------------------------------------------------------
        self.spatial_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=(eeg_channels, 1),
            stride=(1, 1),
            padding=0,
            bias=True,
        )

        self.bn2 = nn.BatchNorm2d(num_filters)
        self.elu2 = nn.ELU()

        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------
        # 4. 1x1 convolution used in the NICE implementation
        #
        # (B, k, 1, T2)
        # -> (B, emb_size, 1, T2)
        # ------------------------------------------------------------
        self.projection = nn.Conv2d(
            in_channels=num_filters,
            out_channels=emb_size,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=True,
        )

        # Calculate the flattened feature dimension.
        temporal_out = input_samples - temporal_kernel + 1
        if temporal_out < pool_kernel:
            raise ValueError(
                "input_samples is too short for the configured "
                "temporal convolution and average pooling."
            )

        pooled_out = math.floor(
            (temporal_out - pool_kernel) / pool_stride
        ) + 1

        self.temporal_out = temporal_out
        self.pooled_out = pooled_out
        self.output_dim = emb_size * pooled_out

    def forward(
        self,
        eeg: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg:
            Tensor of shape (B, C, T).

        attention_mask:
            Optional tensor of shape (B, T).
            1 = valid EEG sample
            0 = padding

            dataset.py already pads with zero, so this mask is not strictly
            required. If supplied, padded samples are explicitly zeroed again.

        Returns
        -------
        torch.Tensor
            Flattened TSConv features with shape (B, output_dim).
        """

        if eeg.ndim != 3:
            raise ValueError(
                f"EEG must have shape (B, C, T), got {tuple(eeg.shape)}"
            )

        batch_size, channels, samples = eeg.shape

        if channels != self.eeg_channels:
            raise ValueError(
                f"Expected {self.eeg_channels} EEG channels, "
                f"but got {channels}."
            )

        if samples != self.input_samples:
            raise ValueError(
                f"Expected EEG length T={self.input_samples}, "
                f"but got T={samples}. "
                "Keep dataset.py collate max_len consistent with "
                "EEGEncoder(input_samples=...)."
            )

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, samples):
                raise ValueError(
                    "attention_mask must have shape "
                    f"(B, T)=({batch_size}, {samples}), "
                    f"but got {tuple(attention_mask.shape)}"
                )

            mask = attention_mask.to(
                device=eeg.device,
                dtype=eeg.dtype,
            )
            eeg = eeg * mask.unsqueeze(1)

        # (B, C, T) -> (B, 1, C, T)
        x = eeg.unsqueeze(1)

        # Temporal Conv
        # (B, 1, C, T)
        # -> (B, k, C, T1)
        x = self.temporal_conv(x)

        # Average Pool
        # -> (B, k, C, T2)
        x = self.avg_pool(x)

        x = self.bn1(x)
        x = self.elu1(x)

        # Spatial Conv
        # -> (B, k, 1, T2)
        x = self.spatial_conv(x)

        x = self.bn2(x)
        x = self.elu2(x)

        x = self.dropout(x)

        # 1x1 projection
        # -> (B, emb_size, 1, T2)
        x = self.projection(x)

        # -> (B, emb_size * T2)
        x = torch.flatten(x, start_dim=1)

        return x


class EEGProjectionHead(nn.Module):
    """
    Project flattened TSConv features into the shared embedding space.

    Default output dimension is 768 so that it matches the output of
    BertSentenceEncoder in text_encoder.py.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 768,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),

            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(projection_dim, projection_dim),
                    nn.Dropout(dropout),
                )
            ),

            nn.LayerNorm(projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGEncoder(nn.Module):
    """
    Complete EEG encoder for the current ChineseEEG project.

    TSConv
        -> flattened EEG feature
        -> projection head
        -> (B, 768)

    No SA / GA.
    No parameters are frozen.
    """

    def __init__(
        self,
        eeg_channels: int,
        input_samples: int = 1500,
        projection_dim: int = 768,
        num_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 51,
        pool_stride: int = 5,
        emb_size: int = 40,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.tsconv = TSConvEncoder(
            eeg_channels=eeg_channels,
            input_samples=input_samples,
            num_filters=num_filters,
            temporal_kernel=temporal_kernel,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            emb_size=emb_size,
            dropout=dropout,
        )

        self.projector = EEGProjectionHead(
            input_dim=self.tsconv.output_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        self.output_dim = projection_dim

        # EEG encoder is trainable by default.
        # This line is not strictly necessary, but makes the intention explicit.
        self.requires_grad_(True)

    def forward(
        self,
        eeg: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input:
            eeg:            (B, C, T)
            attention_mask: (B, T), optional

        Output:
            eeg_embedding:  (B, projection_dim)
        """

        features = self.tsconv(
            eeg=eeg,
            attention_mask=attention_mask,
        )

        eeg_embedding = self.projector(features)

        return eeg_embedding


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Minimal test matching the current dataset.py collate output.
    # ------------------------------------------------------------
    batch_size = 16
    eeg_channels = 128
    max_len = 1500

    eeg = torch.randn(
        batch_size,
        eeg_channels,
        max_len,
    )

    attention_mask = torch.ones(
        batch_size,
        max_len,
        dtype=torch.long,
    )

    encoder = EEGEncoder(
        eeg_channels=eeg_channels,
        input_samples=max_len,
        projection_dim=768,
    )

    embeddings = encoder(
        eeg=eeg,
        attention_mask=attention_mask,
    )

    print("EEG input shape:       ", eeg.shape)
    print("TSConv feature dim:    ", encoder.tsconv.output_dim)
    print("EEG embedding shape:   ", embeddings.shape)
    print(
        "All parameters trainable:",
        all(p.requires_grad for p in encoder.parameters()),
    )

    # Expected with max_len=1500:
    #
    # EEG input shape:        torch.Size([16, 64, 1500])
    # TSConv feature dim:     11440
    # EEG embedding shape:    torch.Size([16, 768])
    # All parameters trainable: True
