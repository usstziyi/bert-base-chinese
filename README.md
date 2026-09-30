# EEG–文本字词对齐

模型使用独立的 EEG Encoder 和 BERT TextEncoder，经两个投影头和 L2 归一化后，
计算双向多正样本对比损失。当前数据单位为高亮字符，每条 EEG 为 `(C, 100)`。
TextEncoder 对每条原始文本按字符数选择层表示：单字取 BERT 最后一层，多字取
最后四层的平均，然后排除特殊 token 与 padding 做 mean pooling，输出仍为 768 维。
混合 batch 中逐条选择。已分词输入可传 `single_char_mask` 明确指定单字样本；
省略时以有效的非特殊 token 数判断，token 数不一定等于原文字符数。

## 正样本定义

Dataset 为相同文本分配相同的 `word_id`；不同位置、文本行或 run 中同一个字
共享标签。`word_id` 在同一个 Dataset 内一致，Subset、shuffle 和 padding
过滤不会重新编号。`word_idx` 表示一次具体出现，不能作为字词身份标签。
合并独立 Dataset 时，需要先统一它们的字词标签映射。

所有 EEG 和文本出现都保留，正样本矩阵为：

```python
positive_mask = word_ids[:, None] == word_ids[None, :]
```

每个 query 对其所有正样本的 `log_softmax` 概率求负平均，随后对 queries 求平均；
EEG→Text 与 Text→EEG 两个方向的损失再取平均。标签全部唯一时，该损失等价于
原来的 CLIP 双向交叉熵。训练会跳过只包含一个字词类别、没有负样本的 batch。

```python
outputs = model(batch['eeg'], batch['text'])
loss = model.contrastive_loss(
    outputs['eeg_embedding'], outputs['text_embedding'], batch['word_id'],
)
loss.backward()
```

输入 EEG 的单位和缩放须与训练配置一致。验证保留整个验证集的所有候选，
分块计算同样的多正样本损失；Recall@1 按 `word_id` 判断，命中同字的另一次
出现也算正确。有重复正样本时，损失的最小值不一定为零；有 k 个正样本时，
该 query 的交叉熵下界为 `log(k)`。跨不同候选库规模比较损失时需要考虑这一点。

## 运行

在项目根目录、已安装依赖的 Python 环境中执行：

```powershell
python -m mind_reading.train --epochs 30 --batch-size 64 --no-include-padding
```

默认冻结 BERT，训练 EEG Encoder 和两个投影头，保留最后一个 run 作验证集。
默认 `--include-padding` 保留补零样本；`--no-include-padding` 同时排除对应
EEG 和文本，包括部分补零和全部补零的片段。输出配置会记录损失类型。

运行测试（无需下载 BERT 权重）：

```powershell
python -m unittest discover -s mind_reading/tests
```
