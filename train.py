# -*- coding: utf-8 -*-
"""
train.py – SCADA × GPT キャプション生成の学習スクリプト
（prompt対応版）
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

from model import SCADADataset, build_caption_model

# ---------------------------------------------------------------------
# 1. 引数定義
# ---------------------------------------------------------------------
def set_default_args(parser: argparse.ArgumentParser):
    parser.add_argument("--train_name_prefix", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--datasets_dpath", type=str, default="./data")
    parser.add_argument("--checkpoints_dpath", type=str, default="./checkpoints")

    parser.add_argument("--rinna_gpt_name", type=str, default="gpt_medium", choices=["gpt_medium", "gpt_1b"])

    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prompt_length", type=int, default=5)
    parser.add_argument("--prefix_dim", type=int, default=512)
    parser.add_argument("--mapping_type", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--only_prefix", action="store_true")

    parser.add_argument("--pretrained_path", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=4)
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=0)

    parser.add_argument("--n_gpu", type=int, default=1)


def make_train_name(args: argparse.Namespace):
    elems = []
    if args.train_name_prefix:
        elems.append(args.train_name_prefix)
    elems.extend([
        args.dataset_name,
        args.rinna_gpt_name,
        args.mapping_type,
        "prefix" if args.only_prefix else "finetune",
        f"ep{args.epochs}",
        f"bs{args.train_batch_size}",
        f"lr{args.lr}",
    ])
    return "-".join(elems)


def save_config(args: argparse.Namespace, out_dir: str):
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


# ---------------------------------------------------------------------
# 3. メイン学習ループ
# ---------------------------------------------------------------------
def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avail_gpu = torch.cuda.device_count()
    args.n_gpu = min(args.n_gpu, avail_gpu)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    pkl_dir = os.path.join(args.datasets_dpath, "processed-scada")
    train_pkl = os.path.join(pkl_dir, "train.pkl")
    valid_pkl = os.path.join(pkl_dir, "valid.pkl")

    train_name = make_train_name(args)
    out_dir = os.path.join(args.checkpoints_dpath, train_name)
    os.makedirs(out_dir, exist_ok=True)
    save_config(args, out_dir)

    train_ds = SCADADataset(train_pkl, args.prefix_length, args.prompt_length)
    valid_ds = SCADADataset(valid_pkl, args.prefix_length, args.prompt_length)
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False)

    model = build_caption_model(
        gpt_variant=args.rinna_gpt_name,
        prefix_length=args.prefix_length,
        prefix_dim=args.prefix_dim,
        mapping_type=args.mapping_type,
        num_layers=args.num_layers,
        only_prefix=args.only_prefix,
        pretrained_path=args.pretrained_path,
    ).to(device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dl) * args.epochs,
    )

    log = []
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1} / {args.epochs} =====")

        model.train()
        train_losses = []
        for tokens, mask, prefix, prompt_tokens in tqdm(train_dl, desc="Train"):
            tokens = tokens.to(device)
            mask = mask.to(device)
            prefix = prefix.to(device, dtype=torch.float32)
            prompt_tokens = prompt_tokens.to(device)

            model.zero_grad()
            outputs = model(tokens=tokens, prefix=prefix, mask=mask, prompt_tokens=prompt_tokens)

            total_prefix_len = args.prefix_length + args.prompt_length
            logits = outputs.logits[:, total_prefix_len - 1 : -1]

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

        avg_train = sum(train_losses) / len(train_losses)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train:.4f}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for tokens, mask, prefix, prompt_tokens in tqdm(valid_dl, desc="Valid"):
                tokens = tokens.to(device)
                mask = mask.to(device)
                prefix = prefix.to(device, dtype=torch.float32)
                prompt_tokens = prompt_tokens.to(device)

                outputs = model(tokens=tokens, prefix=prefix, mask=mask, prompt_tokens=prompt_tokens)

                logits = outputs.logits[:, total_prefix_len - 1 : -1]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tokens.flatten(),
                    ignore_index=0,
                )
                if args.n_gpu > 1:
                    loss = loss.mean()
                val_losses.append(loss.item())

        avg_val = sum(val_losses) / len(val_losses)
        print(f"[Epoch {epoch+1}] Valid Loss: {avg_val:.4f}")

        if args.save_every == 0 or (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(out_dir, f"{epoch+1:03d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] saved checkpoint → {ckpt_path}")

        log.append({"epoch": epoch, "train_loss": avg_train, "valid_loss": avg_val})
        json.dump(log, open(os.path.join(out_dir, "log.json"), "w"), indent=4)

    best_ep = min(log, key=lambda x: x["valid_loss"])["epoch"]
    print(f"Best epoch: {best_ep}")
    return out_dir, os.path.join(out_dir, f"{best_ep:03d}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_default_args(parser)
    args = parser.parse_args()
    train(args)
