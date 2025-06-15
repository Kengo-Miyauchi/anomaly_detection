# -*- coding: utf-8 -*-
"""
train.py – SCADA × GPT キャプション生成の学習スクリプト
===================================================
* 事前に preprocess.py で作った
    └ datasets_dpath/<dataset_name>/processed-scada/{train,valid}.pkl
  を読み込み、CaptionModel を学習する。
"""
import os
import sys
import json
import warnings
import argparse
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

# ===== 新しい model.py から読み込み =====
from model import SCADADataset, build_caption_model

# ---------------------------------------------------------------------
# 1. 引数定義
# ---------------------------------------------------------------------
def set_default_args(parser: argparse.ArgumentParser):
    # データセット／出力
    parser.add_argument("--train_name_prefix", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, required=True)          # 例: scada_coco
    parser.add_argument("--datasets_dpath", type=str, default="./data")     # pkl 群のルート
    parser.add_argument("--checkpoints_dpath", type=str, default="./checkpoints")  # 学習済みモデルの保存先

    # GPT バリアント
    parser.add_argument("--rinna_gpt_name", type=str, default="gpt_medium", choices=["gpt_medium", "gpt_1b"])

    # モデル・学習設定
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_dim", type=int, default=512)
    parser.add_argument("--mapping_type", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--num_layers", type=int, default=4)                # TransformerMapper 用
    parser.add_argument("--only_prefix", action="store_true")               # GPT 本体を凍結

    # 事前学習重み
    parser.add_argument("--pretrained_path", type=str, default=None)

    # 训练ハイパーパラメータ
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=4)
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=0)

    # GPU
    parser.add_argument("--n_gpu", type=int, default=1)

# ---------------------------------------------------------------------
# 2. 補助関数
# ---------------------------------------------------------------------
def make_train_name(args: argparse.Namespace) -> str:
    elems = []
    if args.train_name_prefix:
        elems.append(args.train_name_prefix)
    elems.extend(
        [
            args.dataset_name,
            args.rinna_gpt_name,
            args.mapping_type,
            "prefix" if args.only_prefix else "finetune",
            f"ep{args.epochs}",
            f"bs{args.train_batch_size}",
            f"lr{args.lr}",
        ]
    )
    return "-".join(elems)


def save_config(args: argparse.Namespace, out_dir: str):
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

# ---------------------------------------------------------------------
# 3. メイン学習ループ
# ---------------------------------------------------------------------
def train(args: argparse.Namespace):
    # ---- デバイス設定 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avail_gpu = torch.cuda.device_count()
    if avail_gpu < args.n_gpu:
        warnings.warn(f"Only {avail_gpu} GPU(s) available → n_gpu={avail_gpu}")
        args.n_gpu = avail_gpu

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    print(f"[INFO] #GPUs: {args.n_gpu}  |  train BS (total): {args.train_batch_size}")

    # ---- データパス ----
    pkl_dir = os.path.join(args.datasets_dpath,"processed-scada")
    train_pkl = os.path.join(pkl_dir, "train.pkl")
    valid_pkl = os.path.join(pkl_dir, "valid.pkl")

    # ---- 出力ディレクトリ ----
    train_name = make_train_name(args)
    out_dir = os.path.join(args.checkpoints_dpath, train_name)
    os.makedirs(out_dir, exist_ok=True)
    save_config(args, out_dir)

    # ---- Dataset / DataLoader ----
    train_ds = SCADADataset(train_pkl, prefix_length=args.prefix_length)
    valid_ds = SCADADataset(valid_pkl, prefix_length=args.prefix_length)
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

    # ---- モデル ----
    model = build_caption_model(
        gpt_variant=args.rinna_gpt_name,
        prefix_length=args.prefix_length,
        prefix_dim=args.prefix_dim,
        mapping_type=args.mapping_type,
        num_layers=args.num_layers,
        only_prefix=args.only_prefix,
        pretrained_path=args.pretrained_path,
    ).to(device)

    # DataParallel
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # ---- Optimizer / Scheduler ----
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dl) * args.epochs,
    )

    # ---------------------------------------------------------------
    # 4. Epoch ループ
    # ---------------------------------------------------------------
    log = []
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch} / {args.epochs} =====")
        # ----- Train -----
        model.train()
        train_losses = []
        pbar = tqdm(train_dl, desc="Train")
        for tokens, mask, prefix, _, _ in pbar:
            tokens, mask, prefix = (
                tokens.to(device),
                mask.to(device),
                prefix.to(device, dtype=torch.float32),
            )

            model.zero_grad()
            outputs = model(tokens=tokens, prefix=prefix, mask=mask)
            # logits : [B, P+T, V] → caption 部だけ
            logits = outputs.logits[:, train_ds.prefix_length - 1 : -1]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tokens.flatten(),
                ignore_index=0,
            )
            if args.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = sum(train_losses) / len(train_losses)
        print(f"[Epoch {epoch}] Train Loss: {avg_train:.4f}")

        # ----- Validation -----
        model.eval()
        val_losses = []
        pbar = tqdm(valid_dl, desc="Valid")
        with torch.no_grad():
            for tokens, mask, prefix, _, _ in pbar:
                tokens, mask, prefix = (
                    tokens.to(device),
                    mask.to(device),
                    prefix.to(device, dtype=torch.float32),
                )
                outputs = model(tokens=tokens, prefix=prefix, mask=mask)
                logits = outputs.logits[:, valid_ds.prefix_length - 1 : -1]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tokens.flatten(),
                    ignore_index=0,
                )
                if args.n_gpu > 1:
                    loss = loss.mean()
                val_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val = sum(val_losses) / len(val_losses)
        print(f"[Epoch {epoch}] Valid Loss: {avg_val:.4f}")

        # ----- Checkpoint -----
        if args.save_every == 0 or (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(out_dir, f"{epoch:03d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] saved checkpoint → {ckpt_path}")

        log.append({"epoch": epoch, "train_loss": avg_train, "valid_loss": avg_val})
        json.dump(log, open(os.path.join(out_dir, "log.json"), "w"), indent=4)

    # best checkpoint
    best_ep = min(log, key=lambda x: x["valid_loss"])["epoch"]
    print(f"Best epoch: {best_ep}")
    return out_dir, os.path.join(out_dir, f"{best_ep:03d}.pt")


# ---------------------------------------------------------------------
# 5. エントリポイント
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_default_args(parser)
    args = parser.parse_args()
    train(args)
