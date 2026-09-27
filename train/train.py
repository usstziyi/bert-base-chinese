"""EEG–文本对比学习入口。

在项目根目录运行：python -m train.train --epochs 30 --batch-size 32
也支持直接运行 train/train.py。默认将最后一个 run 留作验证集。
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
import torch.nn.functional as F
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import ChineseEEGDataset, create_eegtext_dataloader
from framework import EEGEncoder, TextEncoder, EEGTextModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--novel-name", default="LittlePrince")
    parser.add_argument("--subject", default="sub-04")
    parser.add_argument("--filtered", default="filtered_0.5_30")
    parser.add_argument("--run-num", type=int, default=7, help="加载第 1 至 N 个 run")
    parser.add_argument("--val-runs", type=int, nargs="+", help="验证 run 编号，从 1 开始；默认最后一个")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--text-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eeg-scale", type=float, default=1e6, help="MNE 的 V 转为 μV；推理须保持一致")
    parser.add_argument("--text-model", default="bert-base-chinese")
    parser.add_argument("--text-max-length", type=int, default=512)
    parser.add_argument("--finetune-text", action="store_true", help="开启 BERT 微调")
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
    for name in ("epochs", "text_max_length", "projection_dim", "projection_hidden_dim", "log_every"):
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
        novel_name=args.novel_name, filtered=args.filtered,
        subject=args.subject, run_num=args.run_num,
    )
    train_indices, val_indices = [], []
    for index, sample in enumerate(dataset.samples):
        target = val_indices if sample["run_idx"] + 1 in args.val_runs else train_indices
        target.append(index)
    if min(len(train_indices), len(val_indices)) < 2:
        raise ValueError("训练集和验证集都至少需要两个 EEG–文本配对样本")

    first = dataset[train_indices[0]]
    n_chans, sfreq = first["eeg"].shape[0], first["sfreq"]
    for run in dataset.eeg_data:
        if not math.isclose(float(run["sfreq"]), sfreq):
            raise ValueError("各 run 采样率必须一致")
        if any(segment.shape[0] != n_chans for segment in run["eeg_segments"]):
            raise ValueError("各 EEG 片段的通道数必须一致")

    options = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = create_eegtext_dataloader(
        Subset(dataset, train_indices), shuffle=True, generator=generator,
        # 单样本 batch 没有负样本，对比损失恒为零；只在尾批为 1 时丢弃。
        drop_last=len(train_indices) % args.batch_size == 1, **options,
    )
    val_loader = create_eegtext_dataloader(Subset(dataset, val_indices), shuffle=False, **options)
    return train_loader, val_loader, n_chans, sfreq


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss, count = 0.0, 0
    for step, batch in enumerate(loader, 1):
        eeg = batch["eeg"].to(device, non_blocking=True) * args.eeg_scale
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(eeg, batch["text"])["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"训练第 {step} 批出现非有限损失")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm, error_if_nonfinite=True,
        )
        optimizer.step()
        total_loss += loss.item() * eeg.size(0)
        count += eeg.size(0)
        if step % args.log_every == 0 or step == len(loader):
            print(f"  step {step}/{len(loader)}  train_loss={total_loss / count:.4f}", flush=True)
    return total_loss / count


@torch.no_grad()
def validate(model, loader, device, args):
    """以整个验证集为候选库，分块计算双向对比损失和 Recall@1。

    正样本是数据集配对索引；相同文本的不同样本仍作为不同候选。
    向量保存在 CPU，避免一次分配完整 N×N 相似度矩阵。
    """
    model.eval()
    eeg_features, text_features = [], []
    for batch in loader:
        eeg = batch["eeg"].to(device, non_blocking=True) * args.eeg_scale
        outputs = model(eeg, batch["text"])
        eeg_features.append(outputs["eeg_embedding"].cpu())
        text_features.append(outputs["text_embedding"].cpu())
    eeg_features = torch.cat(eeg_features)
    text_features = torch.cat(text_features)
    count = len(eeg_features)
    loss_sum, hits = 0.0, []
    for queries, candidates in ((eeg_features, text_features), (text_features, eeg_features)):
        correct = 0
        for start in range(0, count, args.batch_size):
            end = min(start + args.batch_size, count)
            logits = queries[start:end] @ candidates.T / model.temperature
            if not torch.isfinite(logits).all():
                raise FloatingPointError("验证相似度出现非有限值")
            labels = torch.arange(start, end)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
        hits.append(correct / count)
    return {
        "val_loss": loss_sum / (2 * count),
        "eeg_to_text_r1": hits[0], "text_to_eeg_r1": hits[1],
    }


def main():
    args = parse_args()
    # 数据加载器使用相对 data/ 路径，统一以项目根目录为工作目录。
    args.output_dir = args.output_dir.resolve()
    args.text_model = str(Path(args.text_model).resolve()) if Path(args.text_model).is_dir() else args.text_model
    os.chdir(PROJECT_ROOT)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    train_loader, val_loader, n_chans, sfreq = make_loaders(args, device)
    model = EEGTextModel(
        eeg_encoder=EEGEncoder(),
        text_encoder=TextEncoder(
            model_name=args.text_model,
            max_length=args.text_max_length,
        ),
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        freeze_text_encoder=not args.finetune_text,
        temperature=args.temperature,
    ).to(device)
    groups = [{
        "params": [p for name, p in model.named_parameters()
                   if p.requires_grad and not name.startswith("text_encoder.")],
        "lr": args.lr,
    }]
    if args.finetune_text:
        groups.append({"params": list(model.text_encoder.parameters()), "lr": args.text_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    output_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "output_dir": str(output_dir), "n_chans": n_chans, "sfreq": sfreq}
    config["train_runs"] = [run for run in range(1, args.run_num + 1) if run not in args.val_runs]
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"device={device}, channels={n_chans}, sfreq={sfreq}", flush=True)
    print(f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, val_runs={args.val_runs}")
    print(f"保存目录：{output_dir}", flush=True)
    best_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_loss = train_epoch(model, train_loader, optimizer, device, args)
        metrics = {"epoch": epoch, "train_loss": train_loss, **validate(model, val_loader, device, args)}
        metrics["seconds"] = time.perf_counter() - started
        improved = metrics["val_loss"] < best_loss
        best_loss = min(best_loss, metrics["val_loss"])
        checkpoint = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "config": config,
            "metrics": metrics, "best_val_loss": best_loss,
        }
        for name in (["last.pt", "best.pt"] if improved else ["last.pt"]):
            temporary = output_dir / f"{name}.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output_dir / name)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print(
            f"train_loss={train_loss:.4f}  val_loss={metrics['val_loss']:.4f}  "
            f"EEG→Text R@1={metrics['eeg_to_text_r1']:.2%}  "
            f"Text→EEG R@1={metrics['text_to_eeg_r1']:.2%}", flush=True,
        )
    print(f"训练结束，最佳验证损失：{best_loss:.4f}；权重：{output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
