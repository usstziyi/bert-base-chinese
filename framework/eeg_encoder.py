"""EEGNet 特征编码器，供 EEGTextModel 使用。"""

from collections import OrderedDict

import torch
from torch import nn
from braindecode.models import EEGNet


class EEGEncoder(nn.Module):
    """将 (B, C, T) EEG 编码为 (B, feature_dim)。

    保留 EEGNet 时空卷积，去掉分类头，使用全局平均池化。
    共享空间的投影和归一化由 EEGTextModel 完成。
    可接收右侧 padding mask，排除补零区域的归一化统计和平均池化。
    """

    def __init__(self):
        super().__init__()
        n_chans=128
        n_times=1500
        sfreq=256
        F1=8
        D=2
        F2=None
        kernel_length=64
        depthwise_kernel_length=16
        pool1_kernel_size=4
        pool2_kernel_size=8
        pool_mode="mean"
        drop_prob=0.25
        final_conv_length="auto"

        self.n_chans = n_chans
        # EEGNet 对偶数卷积核补齐后会增加一个时间点。
        temporal_extra = int(kernel_length % 2 == 0)
        depthwise_extra = int(depthwise_kernel_length % 2 == 0)
        self.min_samples = max(
            1, pool1_kernel_size * max(1, pool2_kernel_size - depthwise_extra)
            - temporal_extra,
        )
        self.feature_dim = F1 * D if F2 is None else F2 # 16
        eegnet = EEGNet(
            n_chans=n_chans,
            n_outputs=1,
            n_times=n_times,
            sfreq=sfreq,
            # 以下参数均为默认值，此处显式写出以便对照
            F1=F1,
            D=D,
            F2=self.feature_dim,
            kernel_length=kernel_length,
            depthwise_kernel_length=depthwise_kernel_length,
            pool1_kernel_size=pool1_kernel_size,
            pool2_kernel_size=pool2_kernel_size,
            pool_mode=pool_mode,
            drop_prob=drop_prob,
            final_conv_length=final_conv_length,
        )
        self.backbone = nn.Sequential(OrderedDict(
            (name, module)
            for name, module in eegnet.named_children()
            if name != "final_layer"
        ))

    @staticmethod
    def _time_mask(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return (
            torch.arange(features.shape[-1], device=features.device)[None, :] < lengths[:, None]
        )[:, None, None, :]

    def forward(
        self, eeg: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected EEG shape (B, {self.n_chans}, T), got {tuple(eeg.shape)}"
            )
        if eeg.shape[-1] < self.min_samples:
            raise ValueError(f"EEG requires at least {self.min_samples} time samples")

        if attention_mask is None:
            features = self.backbone(eeg) # (B, F2, 1, T)
            return features.mean(dim=(-2, -1)) # (B, feature_dim)=(B, F2)

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

        features = eeg.masked_fill(~attention_mask[:, None, :], 0)
        for layer in self.backbone:
            if isinstance(layer, nn.BatchNorm2d) and layer.training:
                # 将有效位置打包后沿用原 BatchNorm，保留参数、运行统计和 checkpoint 键。
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

            # 使用每个样本独立通过该层时的输出长度，保留卷积的边界 padding 语义。
            if isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d)):
                def time_value(value):
                    return value[-1] if isinstance(value, tuple) else value

                kernel = time_value(layer.kernel_size)
                stride = time_value(layer.stride or layer.kernel_size)
                padding = time_value(layer.padding)
                dilation = time_value(getattr(layer, "dilation", 1))
                lengths = (lengths + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

            # ensuredims 尚未将时间维移动到最后，dimshuffle 之后才可使用时间 mask。
            if features.ndim == 4 and layer is not self.backbone.ensuredims:
                features = features.masked_fill(~self._time_mask(features, lengths), 0)

        return features.sum(dim=(-2, -1)) / (lengths[:, None] * features.shape[2])
