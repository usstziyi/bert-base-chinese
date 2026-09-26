from typing import List

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class BertSentenceEncoder(nn.Module):
    """BERT 句子编码器：取最后 4 层 hidden states 求平均，再按非特殊 token 做 mean pooling。"""

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        device: torch.device = None,
        frozen: bool = True,
        max_length: int = 512,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frozen = frozen
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.feature_dim = self.bert.config.hidden_size
        self.bert.to(device)
        if frozen:
            self.bert.requires_grad_(False)
            self.bert.eval()

    @property
    def device(self) -> torch.device:
        """随父模型 .to(...) 更新设备。"""
        return next(self.bert.parameters()).device

    def train(self, mode: bool = True):
        """父模型调用 .train() 时，保持冻结的 BERT 始终处于 eval 模式（不打开 dropout）。"""
        super().train(mode)
        if not any(param.requires_grad for param in self.bert.parameters()):
            self.bert.eval()
        return self

    def forward(
        self,
        texts: List[str] | str | None = None,
        attention_mask: torch.Tensor | None = None,
        special_tokens_mask: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """文本列表 -> (B, feature_dim)，保留设备和训练梯度。

        也支持通过关键字传入已分词的三个张量。
        """
        if input_ids is None:
            if isinstance(texts, str):
                texts = [texts]
            if not texts or not all(isinstance(text, str) for text in texts):
                raise ValueError("texts must be a non-empty list of strings")
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length, return_special_tokens_mask=True,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            special_tokens_mask = inputs["special_tokens_mask"]
        elif texts is not None:
            raise ValueError("Provide either texts or input_ids, not both")
        if attention_mask is None or special_tokens_mask is None:
            raise ValueError("Tokenized input requires both masks")
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        special_tokens_mask = special_tokens_mask.to(self.device)
        if not any(param.requires_grad for param in self.bert.parameters()):
            self.bert.eval()
            # 冻结模型：不建计算图，省显存
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            # 不冻结模型：建计算图，开显存，开 dropout
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # (4, B, L, H)
        last_four = torch.stack(outputs.hidden_states[-4:], dim=0)
        # 最后 4 层 hidden states 求平均 -> (B, L, H)
        hidden = last_four.mean(dim=0)

        # 排除 [CLS]/[SEP]/[PAD] 等特殊 token，再做 mean pooling
        # (B, L)
        mask = attention_mask * (1 - special_tokens_mask)
        mask = mask.unsqueeze(-1)  # (B, L, 1)
        # (B, L, H) -> (B, H)
        sentence_output = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return sentence_output

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        max_length: int | None = None,
    ) -> torch.Tensor:
        """推理便捷入口：直接传文本列表，返回 (batch_size, hidden_size) 的 CPU 张量。"""
        was_training = self.training
        self.eval()

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length if max_length is None else max_length,
            return_special_tokens_mask=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # (B, H)
        sentence_output = self(
            input_ids=inputs["input_ids"],  # (B, L)
            attention_mask=inputs["attention_mask"],  # (B, L)
            special_tokens_mask=inputs["special_tokens_mask"],  # (B, L)
        )

        if was_training:
            self.train()
        return sentence_output.cpu()


if __name__ == "__main__":
    encoder = BertSentenceEncoder()
    texts = ["今天天气很好", "我们去看电影吧"]
    sentence_output = encoder.encode_texts(texts)
    # (B, H)
    print(sentence_output.shape)  # torch.Size([2, 768])
