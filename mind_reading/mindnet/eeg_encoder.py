"""NICE TSConv 编码器，支持变长 EEG 和右侧 padding。

参考 https://github.com/eeyhsong/NICE-EEG/blob/main/nice_stand.py 。
时间卷积 -> 时间平均池化 -> 空间卷积；用有效时间平均池化替代固定长度
Flatten，共享空间投影和归一化仍由 EEGTextModel 完成。
原 EEGNet 实现保存在 eeg_encoder_eegnet_backup.py。
"""

from collections import OrderedDict

import torch
from torch import nn


class EEGEncoder(nn.Module):
    """将 (B, C, T) EEG 编码为 (B, k)。

    C 是电极数，k 是卷积特征通道数。空间卷积 groups=1，同时融合全部
    电极和输入特征通道。默认适配项目的 128 电极，其余默认参数参考 NICE。
    不包含可选的 SA/GA 模块。
    """

    def __init__(
        self,
        n_chans: int = 128,
        k: int = 40,
        m1: int = 25,
        m2: int = 51,
        s: int = 5,
        drop_prob: float = 0.5,
    ):
        super().__init__()
        for name, value in dict(n_chans=n_chans, k=k, m1=m1, m2=m2, s=s).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= drop_prob <= 1:
            raise ValueError("drop_prob must be between 0 and 1")
        self.n_chans = n_chans
        self.feature_dim = k
        self.min_samples = m1 + m2 - 1
        self.backbone = nn.Sequential(OrderedDict([
            ("conv_temporal", nn.Conv2d(1, k, (1, m1))),
            ("pool_temporal", nn.AvgPool2d((1, m2), (1, s))),
            ("bnorm_temporal", nn.BatchNorm2d(k)),
            ("elu_temporal", nn.ELU()),
            ("conv_spatial", nn.Conv2d(k, k, (n_chans, 1))),
            ("bnorm_spatial", nn.BatchNorm2d(k)),
            ("elu_spatial", nn.ELU()),
            ("dropout", nn.Dropout(drop_prob)),
        ]))

    @staticmethod
    def _time_mask(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return (
            torch.arange(features.shape[-1], device=features.device)[None, :] < lengths[:, None]
        )[:, None, None, :]

    def forward(self, eeg: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected EEG shape (B, {self.n_chans}, T), got {tuple(eeg.shape)}"
            )
        if eeg.shape[-1] < self.min_samples:
            raise ValueError(f"EEG requires at least {self.min_samples} time samples")
        if attention_mask is None:
            # (B, 1, C, T) -> (B, k, 1, L) -> (B, k)
            return self.backbone(eeg.unsqueeze(1)).mean(dim=(-2, -1))
        if attention_mask.shape != (eeg.shape[0], eeg.shape[-1]):
            raise ValueError("EEG attention_mask must have shape (B, T)")
        attention_mask = attention_mask.to(device=eeg.device)
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("EEG attention_mask must contain only 0 or 1")
        attention_mask = attention_mask.bool()
        lengths = attention_mask.sum(dim=-1)
        expected = torch.arange(eeg.shape[-1], device=eeg.device)[None, :] < lengths[:, None]
        if not torch.equal(attention_mask, expected):
            raise ValueError("EEG attention_mask must describe contiguous valid samples followed by right padding")
        if torch.any(lengths < self.min_samples):
            raise ValueError(f"Each EEG sample requires at least {self.min_samples} valid time samples")

        features = eeg.masked_fill(~attention_mask[:, None, :], 0).unsqueeze(1)
        for layer in self.backbone:
            if isinstance(layer, nn.BatchNorm2d) and layer.training:
                # 只用有效位置计算 BN，避免补零污染训练统计量和 running stats。
                valid = self._time_mask(features, lengths)[:, 0].expand(
                    -1, features.shape[2], -1,
                )
                channel_last = features.permute(0, 2, 3, 1)
                packed = channel_last[valid].transpose(0, 1)[None, :, :, None]
                normalized = layer(packed)[0, :, :, 0].transpose(0, 1)
                restored = torch.zeros_like(channel_last)
                restored[valid] = normalized
                features = restored.permute(0, 3, 1, 2)
            else:
                features = layer(features)
            if isinstance(layer, (nn.Conv2d, nn.AvgPool2d)):
                # 无 padding、dilation=1 的输出长度公式，包含向下取整。
                lengths = (lengths - layer.kernel_size[-1]) // layer.stride[-1] + 1
            features = features.masked_fill(~self._time_mask(features, lengths), 0)
        return features.sum(dim=(-2, -1)) / lengths[:, None]
