"""EEG–文本对齐模型。设备由调用方通过 model.to(device) 统一管理。"""

from .eeg_encoder import EEGEncoder
from .text_encoder import TextEncoder
from .eegtext_model import EEGTextModel, ProjectionHead

__all__ = ["EEGEncoder", "TextEncoder", "EEGTextModel", "ProjectionHead"]
