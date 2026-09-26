"""EEGNet 特征编码器，供 EEGTextModel 使用。"""

from collections import OrderedDict

import torch
from torch import nn
from braindecode.models import EEGNet


class EEGEncoder(nn.Module):
    """将 (B, C, T) EEG 编码为 (B, feature_dim)。

    保留 EEGNet 时空卷积，去掉分类头，使用全局平均池化。
    共享空间的投影和归一化由 EEGTextModel 完成。
    当前接口对补零后的完整时间轴池化，不接收有效长度掩码。
    """

    def __init__(
        self,
        n_chans: int = 128,
        n_times: int = 1500,
        sfreq: float = 256,
        F1=8,                # 时间滤波器数量
        D=2,                 # 深度乘数
        F2=None,             # 点卷积滤波器数，None 时内部自动设为 F1*D = 16
        kernel_length=64,    # 第一层时间卷积核长度
        depthwise_kernel_length=16,  # 深度时间卷积核长度
        pool1_kernel_size=4,
        pool2_kernel_size=8,
        pool_mode='mean',
        drop_prob=0.25,
        final_conv_length='auto',
    ):
        super().__init__()
        self.n_chans = n_chans
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

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected EEG shape (B, {self.n_chans}, T), got {tuple(eeg.shape)}"
            )
        # EEGNet 两次池化的默认窗口为 4 和 8。
        if eeg.shape[-1] < 32:
            raise ValueError("EEG requires at least 32 time samples")
        features = self.backbone(eeg)  # (B, F2, 1, T')
        return features.mean(dim=(-2, -1)) # (B, self.feature_dim)=(B, F2)=(B, 16)
        
