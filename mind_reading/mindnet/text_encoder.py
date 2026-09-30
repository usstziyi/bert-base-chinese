from typing import List

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    """BERT 文本编码器：单字取最后一层，多字平均最后四层；调用方控制冻结与模式。"""

    def __init__(self):
        super().__init__()
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.bert = AutoModel.from_pretrained("bert-base-chinese", output_hidden_states=True)
        self.feature_dim = self.bert.config.hidden_size

    @property
    def device(self) -> torch.device:
        """随父模型 .to(...) 更新设备。"""
        return next(self.bert.parameters()).device

    def forward(
        self,
        texts: List[str] | str | None = None,
        attention_mask: torch.Tensor | None = None,
        special_tokens_mask: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        single_char_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """文本列表 -> (B, feature_dim)，保留设备和训练梯度。
        文本 → 分词 → 移动到模型设备 → BERT 编码
            → 按样本选择层表示 → 排除特殊及填充位置 → 文本向量

        字符串按 len(text) == 1 判断单字，混合 batch 中各样本独立选择。
        已分词输入可提供 (B,) bool single_char_mask 指明原文是否单字；
        未提供时以有效的非特殊 token 数是否为 1 判断（无法恢复原文字符数）。
        """
        if input_ids is None:
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(texts, (list, tuple)) or not texts or not all(
                isinstance(text, str) for text in texts
            ):
                raise ValueError("texts must be a non-empty list of strings")
            if single_char_mask is not None:
                raise ValueError("single_char_mask is only supported with input_ids")
            single_char_mask = torch.tensor([len(text) == 1 for text in texts], dtype=torch.bool)

            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.max_length, 
                return_special_tokens_mask=True,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            special_tokens_mask = inputs["special_tokens_mask"]
        elif texts is not None:
            raise ValueError("Provide either texts or input_ids, not both")

        
        if attention_mask is None or special_tokens_mask is None:
            raise ValueError("Tokenized input requires both masks")

        # 移动数据到模型设备
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        special_tokens_mask = special_tokens_mask.to(self.device)

        # 特殊标记和 padding 不参与最终池化，也不计入 token 数。
        mask = attention_mask * (1 - special_tokens_mask)
        if single_char_mask is None:
            single_char_mask = mask.sum(dim=1) == 1
        elif (not isinstance(single_char_mask, torch.Tensor)
              or single_char_mask.shape != (input_ids.shape[0],)
              or single_char_mask.dtype != torch.bool):
            raise ValueError("single_char_mask must be a bool Tensor with shape (B,)")
        single_char_mask = single_char_mask.to(self.device)
        need_layer_average = not single_char_mask.all().item()

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_layer_average,
        )

        hidden = outputs.last_hidden_state  # (B, L, H)，单字直接取最后一层。
        if need_layer_average:
            last_four_mean = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)
            hidden = torch.where(single_char_mask[:, None, None], hidden, last_four_mean)

        # 排除 [CLS]/[SEP]/[PAD] 等特殊 token，再做 mean pooling
        mask = mask.unsqueeze(-1)  # (B, L, 1)
        # (B, L, H) -> (B, H)
        text_features = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return text_features

    
