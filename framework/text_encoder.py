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
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.frozen = frozen
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        if frozen:
            self.bert.requires_grad_(False)
        self.bert.to(device)
        self.bert.eval()

    def train(self, mode: bool = True):
        """父模型调用 .train() 时，保持冻结的 BERT 始终处于 eval 模式（不打开 dropout）。"""
        super().train(mode)
        if self.frozen:
            self.bert.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        """输入为已分词的 batch 张量，返回 (batch_size, hidden_size) 的句子向量。"""
        if self.frozen:
            # 冻结推理：不建计算图，省显存
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # 最后 4 层 hidden states 求平均 -> (batch, seq_len, hidden)
        last_four = torch.stack(outputs.hidden_states[-4:], dim=0)
        hidden = last_four.mean(dim=0)

        # 排除 [CLS]/[SEP]/[PAD] 等特殊 token，再做 mean pooling
        mask = attention_mask * (1 - special_tokens_mask)
        mask = mask.unsqueeze(-1)
        sentence_embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return sentence_embeddings

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> torch.Tensor:
        """推理便捷入口：直接传文本列表，返回 (batch_size, hidden_size) 的 CPU 张量。"""
        was_training = self.training
        self.eval()

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        embeddings = self(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            special_tokens_mask=inputs["special_tokens_mask"],
        )

        if was_training:
            self.train()
        return embeddings.cpu()


if __name__ == "__main__":
    encoder = BertSentenceEncoder()
    texts = ["今天天气很好", "我们去看电影吧"]
    embeddings = encoder.encode_texts(texts)
    print(embeddings.shape)  # torch.Size([2, 768])
