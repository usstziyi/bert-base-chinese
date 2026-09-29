"""用 Transformer 原生 padding mask 编码变长 EEG。

输入 (B, C, T)，输出 (B, d_model)。每个时间点的全部电极线性投影为一个
token，加入动态正弦位置编码后进行时间注意力和有效位置平均池化。
TSConv 原实现（含原有修改）保存在 eeg_encoder_tsconv_backup.py。
"""

import math

import torch
from torch import nn


class EEGEncoder(nn.Module):
    """attention_mask 为 (B, T)，1/True 表示有效，0/False 表示右侧 padding。

    d_model 同时是 Transformer 的 d_model 和输出 feature_dim，必须能被 nhead 整除。
    不传 mask 时所有时间点均有效；每个样本至少需要一个有效时间点。
    使用完整时间分辨率，注意力计算量随 T 平方增长。
    旧 TSConv 的 m1/m2/s 参数及权重仅适用于备份实现。
    """

    def __init__(
        self,
        n_chans: int = 128,
        d_model: int = 64,
        *,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256, # 4*d_model
        drop_prob: float = 0.1,
    ):
        super().__init__()
        # 校验参数是否合法
        for name, value in dict(
            n_chans=n_chans, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not 0 <= drop_prob <= 1:
            raise ValueError("drop_prob must be between 0 and 1")

        self.n_chans = n_chans
        self.feature_dim = d_model
        self.min_samples = 1
        self.input_projection = nn.Linear(n_chans, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,    # Transformer 的特征维度（同时是输入输出维度）
            nhead=nhead,  # 多头注意力的头数
            dim_feedforward=dim_feedforward,  # 前馈神经网络的中间层维度
            dropout=drop_prob,  # Dropout 正则化的概率
            activation="gelu",  # 前馈网络使用的激活函数
            batch_first=True,  # 输入张量形状是否为 (batch_size, seq_len, feature_dim)
            norm_first=True,  # 是否先进行层归一化再执行注意力/前馈计算（Pre-LN）
        )
        self.transformer = nn.TransformerEncoder(
            layer, 
            num_layers=num_layers, 
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        # TransformerEncoder 克隆层结构；分别初始化矩阵，避免各层初始权重相同。
        for encoder_layer in self.transformer.layers:
            for parameter in encoder_layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

    @staticmethod
    def _position_encoding(tokens: torch.Tensor) -> torch.Tensor:
        """动态生成位置编码，不限制最大长度；要求 dimension 为偶数，兼容混合精度。"""
        length, dimension = tokens.shape[1:]
        dtype = torch.float64 if tokens.dtype == torch.float64 else torch.float32
        positions = torch.arange(length, device=tokens.device, dtype=dtype)[:, None] # (T,1)
        frequencies = torch.exp(
            torch.arange(0, dimension, 2, device=tokens.device, dtype=dtype)
            * (-math.log(10000.0) / dimension)
        )
        angles = positions * frequencies # (T, K)，K = d/2
        encoding = torch.empty(length, dimension, device=tokens.device, dtype=dtype)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles.cos()
        return encoding.to(dtype=tokens.dtype)

    def forward(self, eeg: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # 校验EEG输入形状
        if eeg.ndim != 3 or eeg.shape[1] != self.n_chans:
            raise ValueError(f"Expected EEG shape (B, {self.n_chans}, T), got {tuple(eeg.shape)}")
        if eeg.shape[0] == 0 or eeg.shape[-1] < self.min_samples:
            raise ValueError("EEG requires a non-empty batch and at least one time sample")
        
        # 校验attention_mask形状(B,T_max)
        if attention_mask is None:
            attention_mask = torch.ones(eeg.shape[0], eeg.shape[-1], device=eeg.device, dtype=torch.bool)
        else:
            if attention_mask.shape != (eeg.shape[0], eeg.shape[-1]):
                raise ValueError("EEG attention_mask must have shape (B, T)")
            attention_mask = attention_mask.to(device=eeg.device)
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("EEG attention_mask must contain only 0 or 1")
            attention_mask = attention_mask.bool()
        lengths = attention_mask.sum(dim=-1)
        if torch.any(lengths == 0):
            raise ValueError("Each EEG sample requires at least one valid time sample")
        expected = torch.arange(eeg.shape[-1], device=eeg.device)[None, :] < lengths[:, None]
        if not torch.equal(attention_mask, expected):
            raise ValueError("EEG attention_mask must describe contiguous valid samples followed by right padding")

        # PyTorch key_padding_mask 的 True 表示忽略，与数据加载器的语义相反。
        key_padding_mask = ~attention_mask # (B,T_max)
        # 投影前清除 padding，包括 NaN/Inf，避免污染注意力及参数梯度。
        tokens = self.input_projection(
            # 输入：(B, C, T) → (B, T, C)，清零 padding 时间点
            # 经过 input_projection 后：(B, T, d_model)
            eeg.masked_fill(key_padding_mask[:, None, :], 0).transpose(1, 2),
        )
        tokens = tokens + self._position_encoding(tokens)
        tokens = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        # key_padding_mask 屏蔽 key/value，但不清零 padding query 的输出。
        # 把(B,T_max,d_model) 里 padding 时刻的整行清零，使求和/求平均只发生在有效时间点上。
        tokens = tokens.masked_fill(key_padding_mask[:, :, None], 0)
        return tokens.sum(dim=1) / lengths[:, None].to(dtype=tokens.dtype)
