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
import pickle
import pandas as pd
import csv
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer

from model import SCADADataset, build_caption_model

# ---------------------------------------------------------------------
# 1. 引数定義
# ---------------------------------------------------------------------
def set_default_args(parser: argparse.ArgumentParser):
    parser.add_argument("--train_name_prefix", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--datasets_dpath", type=str, default="./data")
    parser.add_argument("--checkpoints_dpath", type=str, default="./checkpoints")
    parser.add_argument("--prompt_file", type=str, default="./attributes.csv")

    parser.add_argument("--llm_name", type=str, default="gpt_medium", choices=["gpt_medium", "gpt_1b", "llama3"])

    parser.add_argument("--prefix_length", type=int, default=10)
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

# ---------------------------------------------------------------------
# 2. 補助関数
# ---------------------------------------------------------------------
def make_train_name(args: argparse.Namespace):
    elems = []
    if args.train_name_prefix:
        elems.append(args.train_name_prefix)
    elems.extend([
        args.dataset_name,
        args.llm_name,
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
def make_prompt_from_csv(csv_path: str, top_k: int = 40) -> str:
    # 属性CSVの1行目からカラム名を取得してプロンプトを作る
    df = pd.read_csv(csv_path, nrows=1)
    column_names = list(df.columns)
    selected = column_names[:top_k]
    prompt = "以下の語句のいずれかを必ず含んで、異常の説明文を生成してください：\n"
    prompt += "".join(f"・{name}\n" for name in selected)
    prompt += "\n→ "
    return prompt

def load_data_with_prompt(pkl_path: str, prompt: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for cap in data["captions"]:
        cap["prompt"] = prompt
    return data

# train()関数内の、pkl読み込み部分を以下のように置き換え
def train(args: argparse.Namespace):
    # ---- デバイス設定 ----Add commentMore actions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avail_gpu = torch.cuda.device_count()
    if avail_gpu < args.n_gpu:
        warnings.warn(f"Only {avail_gpu} GPU(s) available → n_gpu={avail_gpu}")
        args.n_gpu = avail_gpu

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    print(f"[INFO] #GPUs: {args.n_gpu}  |  train BS (total): {args.train_batch_size}")

    pkl_dir = os.path.join(args.datasets_dpath, "processed-scada")
    train_pkl = os.path.join(pkl_dir, "train.pkl")
    valid_pkl = os.path.join(pkl_dir, "valid.pkl")
    attributes_csv = os.path.join(args.datasets_dpath, "attributes.csv")  # CSVパスを引数などで指定してもよい

    # 属性CSVからprompt作成
    prompt_text = make_prompt_from_csv(attributes_csv, top_k=40)

    # train/validデータ読み込み時にprompt追加
    train_data = load_data_with_prompt(train_pkl, prompt_text)
    valid_data = load_data_with_prompt(valid_pkl, prompt_text)

    # Datasetは元のpklではなく、prompt付きの辞書を受け取るように仮に変更するため、
    # SCADADatasetを少し修正するか、元のpklに一時的に保存する必要あり。
    # ここでは、pickleを一時保存する例を示します：

    train_prompted_pkl = os.path.join(pkl_dir, "train_with_prompt.pkl")
    valid_prompted_pkl = os.path.join(pkl_dir, "valid_with_prompt.pkl")

    with open(train_prompted_pkl, "wb") as f:
        pickle.dump(train_data, f)
    with open(valid_prompted_pkl, "wb") as f:
        pickle.dump(valid_data, f)
    train_ds = SCADADataset(data_path=train_prompted_pkl, prefix_length=args.prefix_length, prompt_length=args.prompt_length)
    valid_ds = SCADADataset(data_path=valid_prompted_pkl, prefix_length=args.prefix_length, prompt_length=args.prompt_length)
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False)

    model = build_caption_model(
        gpt_variant=args.llm_name,
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
        for tokens, mask, prefix, prompt_tokens, _ in tqdm(train_dl, desc="Train"):
            tokens = tokens.to(device)
            mask = mask.to(device)  # これはテキスト部分のmask
            prefix = prefix.to(device, dtype=torch.float32)
            prompt_tokens = prompt_tokens.to(device) if prompt_tokens is not None else None

            batch_size = tokens.size(0)
            prefix_len = prefix.size(1)  # prefixは[batch, prefix_dim]なので要注意。prefixの長さはモデルのprefix_lengthで固定
            # prefix_lenはモデルの prefix_length を使うのが正しい。ここは args.prefix_length か model.prefix_length
            prefix_len = args.prefix_length if hasattr(args, "prefix_length") else 10  # デフォルト10など

            prompt_len = prompt_tokens.size(1) if (prompt_tokens is not None and prompt_tokens.dim() == 2) else 0
            seq_len = tokens.size(1)

            # prefix_mask と prompt_mask を作成
            prefix_mask = torch.ones(batch_size, prefix_len, device=device)
            prompt_mask = torch.ones(batch_size, prompt_len, device=device) if prompt_len > 0 else torch.zeros(batch_size, 0, device=device)

            # full_mask = prefix + prompt + text_mask
            full_mask = torch.cat([prefix_mask, prompt_mask, mask], dim=1)  # shape: [B, prefix_len + prompt_len + seq_len]

            model.zero_grad()
            outputs = model(tokens=tokens, prefix=prefix, mask=full_mask, prompt_tokens=prompt_tokens)

            # prefix_length + prompt_length の計算も args.prompt_length などを使用
            total_prefix_len = prefix_len + prompt_len
            logits = outputs.logits[:, total_prefix_len - 1 : -1]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tokens.flatten(),
                ignore_index=tokenizer.pad_token_id,
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
            for tokens, mask, prefix, prompt_tokens, _ in tqdm(valid_dl, desc="Valid"):
                tokens = tokens.to(device)
                mask = mask.to(device)  # テキスト部分のマスク
                prefix = prefix.to(device, dtype=torch.float32)
                prompt_tokens = prompt_tokens.to(device) if prompt_tokens is not None else None

                batch_size = tokens.size(0)
                prefix_len = args.prefix_length if hasattr(args, "prefix_length") else 10
                prompt_len = prompt_tokens.size(1) if (prompt_tokens is not None and prompt_tokens.dim() == 2) else 0
                seq_len = tokens.size(1)

                prefix_mask = torch.ones(batch_size, prefix_len, device=device)
                prompt_mask = torch.ones(batch_size, prompt_len, device=device) if prompt_len > 0 else torch.zeros(batch_size, 0, device=device)
                full_mask = torch.cat([prefix_mask, prompt_mask, mask], dim=1)  # [B, prefix_len + prompt_len + seq_len]

                outputs = model(tokens=tokens, prefix=prefix, mask=full_mask, prompt_tokens=prompt_tokens)

                total_prefix_len = prefix_len + prompt_len
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
            ckpt_path = os.path.join(args.checkpoints_dpath, f"{epoch+1:03d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] saved checkpoint → {ckpt_path}")

        log.append({"epoch": epoch, "train_loss": avg_train, "valid_loss": avg_val})
        json.dump(log, open(os.path.join(args.checkpoints_dpath, "log.json"), "w"), indent=4)

    best_ep = min(log, key=lambda x: x["valid_loss"])["epoch"]
    print(f"Best epoch: {best_ep}")
    return args.checkpoints_dpath, os.path.join(args.checkpoints_dpath, f"{best_ep:03d}.pt")


def load_attributes(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        attrs = next(reader)
    return attrs

def make_prompt_from_attrs(attr_list):
    lines = [f"・{attr}" for attr in attr_list]
    prompt_text = (
        "以下の語句のいずれかを必ず含んで、異常の説明文を生成してください：\n"
        + "\n".join(lines)
        + "\n\n→ "
    )
    return prompt_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_default_args(parser)

    # 追加引数：属性CSVファイルパス
    parser.add_argument("--attributes_csv", type=str, required=True, help="属性情報のCSVファイルパス")

    args = parser.parse_args()

    # プロンプト文を作成してトークナイズし長さを計算
    prompt_text = make_prompt_from_attrs(load_attributes(args.attributes_csv))
    tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5", use_fast=False)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    if prompt_tokens[0] == tokenizer.bos_token_id:
        prompt_tokens = prompt_tokens[1:]
    args.prompt_length = len(prompt_tokens)

    print(f"[INFO] Generated prompt with {args.prompt_length} tokens.")

    # 以降、train関数へ渡して学習を開始
    train(args)

