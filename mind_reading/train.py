"""EEG–文本对比学习入口。

在项目根目录运行：python -m mind_reading.train --epochs 30 --batch-size 32
也支持直接运行 mind_reading/train.py。默认将最后一个 run 留作验证集。
保留重复字词样本，同 word_id 的全部跨模态候选均为正样本。
每轮更新结束后，同一 checkpoint 用 eval() 分别评估训练集和验证集的 batch loss。
编码器结构、BERT 模型和文本最大长度由各 encoder 内部定义。
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


if __package__:
    from .dataset import ChineseEEGDataset, create_eegtext_dataloader
    from .mindnet import EEGEncoder, TextEncoder, EEGTextModel
else:
    from dataset import ChineseEEGDataset, create_eegtext_dataloader
    from mindnet import EEGEncoder, TextEncoder, EEGTextModel



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--novel-name", default="LittlePrince")
    parser.add_argument("--subject", default="sub-04")
    parser.add_argument("--filtered", default="filtered_0.5_30")
    parser.add_argument("--run-num", type=int, default=7, help="加载第 1 至 N 个 run")
    parser.add_argument("--include-padding", action=argparse.BooleanOptionalAction, default=True,
                        help="保留含补零点的 EEG–文本配对（用 --no-include-padding 排除）")
    parser.add_argument("--val-runs", type=int, nargs="+", help="验证 run 编号，从 1 开始；默认最后一个")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--text-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eeg-scale", type=float, default=1e6, help="MNE 的 V 转为 μV；推理须保持一致")
    parser.add_argument("--freeze-text-encoder", action=argparse.BooleanOptionalAction, default=True,
                        help="冻结 BERT（默认冻结，用 --no-freeze-text-encoder 解冻）")
    parser.add_argument("--freeze-eeg-encoder", action=argparse.BooleanOptionalAction, default=False,
                        help="冻结 EEG 编码器（默认不冻结）")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-hidden-dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda:0 等")
    parser.add_argument("--num-workers", type=int, default=0, help="Windows 下默认 0，避免复制预加载数据")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.run_num < 2 or args.batch_size < 2:
        parser.error("run-num 和 batch-size 至少为 2")
    for name in ("epochs", "projection_dim", "projection_hidden_dim", "log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须大于 0")
    for name in ("lr", "text_lr", "max_grad_norm", "eeg_scale", "temperature"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} 必须为有限正数")
    if args.num_workers < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("num-workers 和 weight-decay 必须非负")
    args.val_runs = sorted(set(args.val_runs or [args.run_num]))
    if any(run < 1 or run > args.run_num for run in args.val_runs) or len(args.val_runs) == args.run_num:
        parser.error("val-runs 必须在 1..run-num 内，并至少保留一个训练 run")
    return args


def make_loaders(args, device):
    dataset = ChineseEEGDataset(
        novel_name=args.novel_name, 
        filtered=args.filtered,
        subject=args.subject, 
        run_num=args.run_num,
        include_padding=args.include_padding,
    )
    train_indices, val_indices = [], []
    for index, sample in enumerate(dataset.samples):
        target = val_indices if sample["run_idx"] + 1 in args.val_runs else train_indices
        target.append(index)
        
    if min(len(train_indices), len(val_indices)) < 2:
        raise ValueError("训练集和验证集都至少需要两个 EEG–文本配对样本")

    first = dataset[train_indices[0]]
    n_chans = first["eeg"].shape[0]
    for word in dataset.eeg_words:
        if word["eeg"].shape[0] != n_chans:
            raise ValueError("各 EEG 片段的通道数必须一致")

    options = dict(
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda", # 只在 CUDA 上使用 pin_memory
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = create_eegtext_dataloader(
        Subset(dataset, train_indices), 
        shuffle=True, 
        generator=generator,
        # 单样本 batch 没有负样本，对比损失恒为零；只在尾批为 1 时丢弃。
        drop_last=len(train_indices) % args.batch_size == 1, 
        **options
    )
    val_loader = create_eegtext_dataloader(
        Subset(dataset, val_indices), 
        shuffle=False, 
        **options
    )
    return train_loader, val_loader, n_chans


def make_batch_eval_loader(source_loader, *, batch_size, seed):
    """一次性固定随机分组；不消费训练 loader 的随机状态。

    两个 split 都使用完整 batch，保证候选数量相同。复用 source 的数据与
    collate_fn，支持嵌套 Subset；不复制 EEG，不改变原 loader 的采样顺序。
    """
    if batch_size < 2 or len(source_loader.dataset) < batch_size:
        raise ValueError("Batch evaluation requires batch_size >= 2 and at least one full batch per split")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(source_loader.dataset), generator=generator).tolist()
    return DataLoader(
        Subset(source_loader.dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=source_loader.num_workers,
        pin_memory=source_loader.pin_memory,
        collate_fn=source_loader.collate_fn,
        generator=generator,
    )


@torch.no_grad()
def evaluate_batch_loss(model, loader, device, args, *, split_name="eval"):
    """冻结当前权重、关闭 dropout，以 batch 内候选计算多正样本损失。

    输入为固定分组且 drop_last=True 的 loader。按有效样本数加权平均，
    与训练一样跳过没有不同字负样本的 batch；报告尾部及跳过的样本数量。
    """
    model.eval()
    loss_sum, count, skipped_batches = 0.0, 0, 0
    for step, batch in enumerate(loader, 1):
        if batch["word_id"].unique().numel() < 2:
            skipped_batches += 1
            continue
        eeg = batch["eeg"].to(device, non_blocking=True) * args.eeg_scale
        outputs = model(eeg, batch["text"])
        loss = model.contrastive_loss(
            outputs["eeg_embedding"], outputs["text_embedding"], batch["word_id"],
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{split_name} batch evaluation produced a non-finite loss")
        loss_sum += loss.item() * eeg.size(0)
        count += eeg.size(0)
        if step == 1 or step == len(loader) or step % getattr(args, "log_every", 20) == 0:
            print(f"  {split_name} eval batch {step}/{len(loader)}  loss={loss_sum / count:.4f}", flush=True)
    if count == 0:
        raise ValueError(f"{split_name}: no evaluation batches contain at least two distinct word_ids")
    return {
        "loss": loss_sum / count,
        "samples": count,
        "skipped_batches": skipped_batches,
        "excluded_samples": len(loader.dataset) - count,
    }


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss, count, skipped_batches = 0.0, 0, 0
    for step, batch in enumerate(loader, 1):
        # 单一字词 batch 没有负样本，不用于辨别字词身份的训练。
        if batch["word_id"].unique().numel() < 2:
            skipped_batches += 1
            continue
        eeg = batch["eeg"].to(device, non_blocking=True) * args.eeg_scale
        # 文本是字符串列表；分词后的张量由 TextEncoder 移到模型设备。
        text = batch["text"]
        optimizer.zero_grad(set_to_none=True)
        outputs = model(eeg, text)
        loss = model.contrastive_loss(
            outputs["eeg_embedding"], outputs["text_embedding"], batch["word_id"],
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"训练第 {step} 批出现非有限损失")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm, error_if_nonfinite=True,
        )
        optimizer.step()
        total_loss += loss.item() * eeg.size(0)
        count += eeg.size(0)
        print(f"  step {step}/{len(loader)}  train_loss={total_loss / count:.4f}", flush=True)
        # if step % args.log_every == 0 or step == len(loader):
        #     print(f"  step {step}/{len(loader)}  train_loss={total_loss / count:.4f}", flush=True)
    if count == 0:
        raise ValueError("No training batches contain at least two distinct word_ids")
    if skipped_batches:
        print(f"跳过 {skipped_batches} 个不含不同字词负样本的 batch", flush=True)
    return total_loss / count


@torch.no_grad()
def validate(model, loader, device, args):
    """以整个验证集为候选库，分块计算双向对比损失和 Recall@1。
    保留全部候选；同 word_id 的所有出现均为正样本，Recall@1 按字词身份计分。
    向量保存在 CPU，避免一次分配完整 N×N 相似度矩阵。
    """
    model.eval()
    eeg_features, text_features, word_ids = [], [], []
    for i, batch in enumerate(loader):
        eeg = batch["eeg"].to(device, non_blocking=True) * args.eeg_scale
        text = batch["text"]
        outputs = model(eeg, text)
        eeg_features.append(outputs["eeg_embedding"].cpu())
        text_features.append(outputs["text_embedding"].cpu())
        word_ids.append(batch["word_id"].cpu())
        print(f"val step {i}/{len(loader)}")
    eeg_features = torch.cat(eeg_features) # (N, D)
    text_features = torch.cat(text_features) # (N, D)
    word_ids = torch.cat(word_ids) # 全 Dataset 一致的标签，跨 batch 同字仍为正样本。
    count = len(eeg_features) # N
    loss_sum, hits = 0.0, []
    for queries, candidates in ((eeg_features, text_features), (text_features, eeg_features)):
        correct = 0
        for start in range(0, count, args.batch_size):
            end = min(start + args.batch_size, count)
            logits = queries[start:end] @ candidates.T / model.temperature
            if not torch.isfinite(logits).all():
                raise FloatingPointError("验证相似度出现非有限值")
            positive_mask = word_ids[start:end, None] == word_ids[None, :]
            loss_sum += model.multi_positive_loss(logits, positive_mask).item() * (end - start)
            predicted_ids = word_ids[logits.argmax(dim=1)]
            correct += (predicted_ids == word_ids[start:end]).sum().item()
        hits.append(correct / count)
    return {
        "val_loss": loss_sum / (2 * count),
        "eeg_to_text_r1": hits[0], 
        "text_to_eeg_r1": hits[1],
    }


def main():
    args = parse_args()
    print(f"freeze_text_encoder={args.freeze_text_encoder}")
    print(f"freeze_eeg_encoder={args.freeze_eeg_encoder}")

    # 数据加载器使用相对 data/ 路径，统一以项目根目录为工作目录。
    args.output_dir = args.output_dir.resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    train_loader, val_loader, n_chans = make_loaders(args, device)
    train_eval_loader = make_batch_eval_loader(train_loader, batch_size=args.batch_size, seed=args.seed)
    val_eval_loader = make_batch_eval_loader(val_loader, batch_size=args.batch_size, seed=args.seed + 1)
    model = EEGTextModel(
        eeg_encoder=EEGEncoder(n_chans=n_chans),
        text_encoder=TextEncoder(),
        freeze_text_encoder=args.freeze_text_encoder,
        freeze_eeg_encoder=args.freeze_eeg_encoder,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        temperature=args.temperature,
    ).to(device)

    # 给非文本编码器参数添加学习率组
    groups = [{
        "params": [p for name, p in model.named_parameters()
                   if p.requires_grad and not name.startswith("text_encoder.")],
        "lr": args.lr,
    }]
    # 单独为文本编码器参数添加学习率组
    if not args.freeze_text_encoder:
        groups.append({"params": list(model.text_encoder.parameters()), "lr": args.text_lr})
    # 初始化优化器
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    # config.json
    output_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "output_dir": str(output_dir), "n_chans": n_chans}
    config["text_model"] = model.text_encoder.tokenizer.name_or_path
    config["text_max_length"] = model.text_encoder.max_length
    config["train_runs"] = [run for run in range(1, args.run_num + 1) if run not in args.val_runs]
    config["contrastive_objective"] = "symmetric_multi_positive_word_identity"
    config["batch_evaluation"] = {
        "model_mode": "eval",
        "batch_size": args.batch_size,
        "train_seed": args.seed,
        "val_seed": args.seed + 1,
        "drop_last": True,
        "skip_single_word_batches": True,
    }
    config["checkpoint_selection_metric"] = "val_batch_loss"
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"device={device}, channels={n_chans}", flush=True)
    print(f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, val_runs={args.val_runs}")
    print(f"保存目录：{output_dir}", flush=True)

    best_loss = math.inf # 正无穷大
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train_epoch(model, train_loader, optimizer, device, args)
        # 三次评估之间不更新参数，全部使用该轮结束时的同一份权重。
        train_batch_metrics = evaluate_batch_loss(model, train_eval_loader, device, args, split_name="train")
        val_batch_metrics = evaluate_batch_loss(model, val_eval_loader, device, args, split_name="val")
        validation_metrics = validate(model, val_loader, device, args)
        metrics = {
            "epoch": epoch, 
            "train_loss": train_loss, 
            **{f"train_eval_batch_{key}": value for key, value in train_batch_metrics.items()},
            **{f"val_batch_{key}": value for key, value in val_batch_metrics.items()},
            **validation_metrics,
            "seconds": time.perf_counter() - started,
        }
        improved = metrics["val_batch_loss"] < best_loss
        best_loss = min(best_loss, metrics["val_batch_loss"])
        checkpoint = {
            "epoch": epoch, 
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), 
            "config": config,
            "metrics": metrics, 
            "best_val_batch_loss": best_loss,
        }
        for name in (["last.pt", "best.pt"] if improved else ["last.pt"]):
            temporary = output_dir / f"{name}.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output_dir / name)
            
        # history.jsonl
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print(
            f"train_loss={train_loss:.4f}  "
            f"train_eval_batch_loss={metrics['train_eval_batch_loss']:.4f}  "
            f"val_batch_loss={metrics['val_batch_loss']:.4f}  "
            f"val_full_loss={metrics['val_loss']:.4f}  "
            f"full EEG→Text R@1={metrics['eeg_to_text_r1']:.2%}  "
            f"full Text→EEG R@1={metrics['text_to_eeg_r1']:.2%}", flush=True,
        )
        print(
            f"batch eval samples: train={metrics['train_eval_batch_samples']} "
            f"(excluded={metrics['train_eval_batch_excluded_samples']}), "
            f"val={metrics['val_batch_samples']} (excluded={metrics['val_batch_excluded_samples']})", flush=True,
        )
    print(f"训练结束，最佳 val_batch_loss：{best_loss:.4f}；权重：{output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
