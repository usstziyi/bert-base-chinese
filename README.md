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

## 同一 checkpoint 的 batch 评估

每轮参数更新结束后，用同一份模型权重、`model.eval()` 和 `torch.no_grad()`
分别评估训练集和验证集。两个 split 都使用与训练相同的 batch size，在各自
batch 内计算双向多正样本损失；不反向传播或更新参数。评估分组只在运行开始时
随机生成一次，训练评估 seed 为 `--seed`，验证评估为 `--seed + 1`，每轮保持一致，
并使用独立的随机生成器，避免消费训练 loader 的 shuffle 随机状态。

日志和 checkpoint metrics 中的字段含义：

| 字段 | 含义 |
| --- | --- |
| `train_loss` | 参数更新过程中累计的训练损失，训练模式开启 dropout |
| `train_eval_batch_loss` | 本轮结束的固定权重，在训练集上以 eval 模式计算的 batch loss |
| `val_batch_loss` | 同一权重，在验证集上以 eval 模式计算的 batch loss |
| `val_loss` | 全验证集作为候选库的损失，控制台标为 `val_full_loss` |
| `eeg_to_text_r1` / `text_to_eeg_r1` | 全验证候选库上的字词身份 Recall@1 |

判断 batch 级泛化时，比较 `train_eval_batch_loss` 和 `val_batch_loss`。
`best.pt` 按最低 `val_batch_loss` 选择，checkpoint 保存 `best_val_batch_loss`；
`last.pt` 保存最新轮次。配置中记录评估方式和 checkpoint 选择指标。
旧日志的 `val_loss` 仍然是全库损失，不能直接与新的 `val_batch_loss` 比较。

两套 batch 评估都排除不足一批的尾部，保证候选数量一致，并跳过仅含一种字词
的 batch；`*_samples`、`*_excluded_samples` 和 `*_skipped_batches` 记录有效样本、
排除样本和跳过批次数。每个 split 至少要能组成一个完整且含两种字词的 batch，
否则评估报错；数据较少时需降低 batch size。新增评估会增加每轮运行时间。

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
