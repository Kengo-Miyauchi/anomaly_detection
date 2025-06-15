# -*- coding: utf-8 -*-
"""
inference.py - SCADA ベクトル → GPT2 で説明文を生成
=================================================
使い方例
--------
python inference.py \
    --scada_csv ./sample/001.csv \
    --tabnet_ckpt ./mnt/iot-qnap5/model/haenkaze/tabnet-pretrain-out2023-40dim/pretrained.pth \
    --cap_ckpt  ./ckpt/scada-run-ep10-bs8-lr2e-05/009.pt
"""
import argparse
import numpy as np
import torch
from transformers import T5Tokenizer
import os, json

from model import CaptionModel, build_caption_model   # ← 新しい model.py 由来
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# 0. TabNet Encoder をロードするユーティリティ
# ----------------------------------------------------------------------
def load_tabnet_encoder(ckpt_path: str):
    """
    ckpt に保存された `unsupervised_model` から encoder を取り出す想定。
    """
    full_model = torch.load(ckpt_path, map_location=DEVICE)
    # 多くの場合 `unsupervised_model` という属性名で保存されている
    encoder = (
        full_model.network.encoder
        if hasattr(full_model, "network")
        else getattr(full_model, "encoder", full_model)
    )
    encoder.eval().to(DEVICE)
    return encoder


# ----------------------------------------------------------------------
# 1. Predictor クラス
# ----------------------------------------------------------------------
class Predictor:
    """
    TabNet で抽出した `prefix_vec` を CaptionModel に渡して
    ビームサーチで説明文を生成するユーティリティ
    """

    def __init__(
        self,
        cap_model: CaptionModel,
        cap_tokenizer: T5Tokenizer,
        tabnet_encoder,
        device: torch.device = DEVICE,
    ):
        self.cap_model = cap_model.eval().to(device)
        self.tokenizer = cap_tokenizer
        self.stop_token = cap_tokenizer.eos_token
        self.encoder = tabnet_encoder  # TabNet Encoder
        self.device = device

    # --------------------------------------------------------------
    # 1-1. SCADA CSV をロードして TabNet Encoder → prefix (Tensor[1,prefix_dim])
    # --------------------------------------------------------------
    def _encode_scada(self, csv_path: str) -> torch.Tensor:
        import pandas as pd
        from util_module.extract_features import create_dataloader

        df = pd.read_csv(csv_path, skiprows=1)
        series = df.values  # [T,D]

        dataloader = create_dataloader(series, batch_size=128, need_shuffle=False)

        features = []
        for batch in dataloader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                step_outputs, _ = self.encoder(batch)
            encoder_out = sum(step for step in step_outputs).cpu()
            features.append(encoder_out)

        embedding = torch.cat(features).mean(dim=0, keepdim=True)  # [1,D]
        return embedding.to(self.device, dtype=torch.float32)      # 最後にGPUへ



    # --------------------------------------------------------------
    # 1-2. パブリック API
    # --------------------------------------------------------------
    def caption(self, csv_path: str, time_range: str, beam_size: int = 5, max_len: int = 64, temperature: float = 1.0):
        prefix_vec = self._encode_scada(csv_path)                          # [1,prefix_dim]
        if time_range is not None:
            prompt = f"{time_range}のデータに基づいて: "
        else:
            prompt = ""

        with torch.no_grad():
            prefix_embed = self.cap_model.prefix_mapper(prefix_vec)        # [1,P,E]
            prefix_embed = prefix_embed.view(1, self.cap_model.prefix_length, -1)
        captions = self._generate_beam(
            embed=prefix_embed,
            beam_size=beam_size,
            prompt=prompt,
            entry_length=max_len,
            temperature=temperature,
        )
        return captions[0]  # 最高スコア 1 件を返す

    # --------------------------------------------------------------
    # 1-3. ビームサーチ (ClipCap 実装を踏襲)
    # --------------------------------------------------------------
    def _generate_beam(self, embed, beam_size, prompt=None, entry_length=67, temperature=1.0):
        stop_idx = self.tokenizer.encode(self.stop_token)[0]
        device = self.device

        scores = None
        seq_len = torch.ones(beam_size, device=device)
        is_stop = torch.zeros(beam_size, device=device, dtype=torch.bool)

        if prompt is not None:
            prompt_tokens = torch.tensor(self.tokenizer.encode(prompt), device=device).unsqueeze(0)  # [1, T]
            prompt_embed = self.cap_model.gpt.transformer.wte(prompt_tokens)  # [1, T, E]
            prefix_embed = embed  # [1, P, E]
            generated = torch.cat((prefix_embed, prompt_embed), dim=1)  # [1, P+T, E]

            # --- tokens（prefix部をダミーで埋める）---
            dummy_prefix_tokens = torch.zeros(1, self.cap_model.prefix_length, dtype=torch.long, device=device)
            tokens = torch.cat((dummy_prefix_tokens, prompt_tokens), dim=1)  # [1, P+T]
        else:
            generated = embed  # [1, P, E]
            tokens = torch.zeros(1, self.cap_model.prefix_length, dtype=torch.long, device=device)  # [1, P]

        # Expand for beam size
        generated = generated.expand(beam_size, *generated.shape[1:])
        tokens = tokens.expand(beam_size, *tokens.shape[1:])


        for _ in range(entry_length):
            outputs = self.cap_model.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            # 1ステップ目（最初のトークン予測）
            if scores is None:
                logits = logits[0]                           # shape: [vocab_size]
                scores, next_tok = logits.topk(beam_size)    # shape: [beam_size]
                next_tok = next_tok.unsqueeze(1)             # [beam_size, 1]
                tokens = next_tok.clone()
                generated = generated.expand(beam_size, *generated.shape[1:])  # [beam, T_prompt, E]
            else:
                logits[is_stop] = -float("inf")
                logits[is_stop, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_len[~is_stop] += 1
                scores_norm = scores_sum / seq_len[:, None]
                scores_norm = scores_norm.view(-1)
                scores, idx = scores_norm.topk(beam_size, -1)
                next_src = idx // logits.size(1)
                next_tok = (idx % logits.size(1)).unsqueeze(1)
                tokens = tokens[next_src]
                tokens = torch.cat([tokens, next_tok], 1)
                generated = generated[next_src]
                seq_len = seq_len[next_src]
                is_stop = is_stop[next_src]
                scores = scores * seq_len  # undo normalization

                tok_embed = self.cap_model.gpt.transformer.wte(next_tok.squeeze(1))  # [B, E]
                tok_embed = tok_embed.unsqueeze(1)                                   # [B, 1, E]
                generated = torch.cat([generated, tok_embed], dim=1)                 # [B, T+1, E]


                # next_tok: shape should be [beam_size, 1]
                if next_tok.dim() > 1:
                    next_tok_squeezed = next_tok.squeeze(1)
                else:
                    next_tok_squeezed = next_tok  # already squeezed

                # 比較：eq → shape [beam_size]
                stop_flags = next_tok_squeezed.eq(stop_idx)

                # 論理和（is_stop: [beam_size]）
                is_stop = is_stop | stop_flags


            if is_stop.all():
                break

        scores = scores / seq_len
        outputs = tokens.cpu().numpy()
        decoded = [self.tokenizer.decode(o[: int(l)]) for o, l in zip(outputs, seq_len)]
        order = scores.argsort(descending=True)
        return [decoded[i] for i in order]


# ----------------------------------------------------------------------
# 2. エントリポイント
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scada_csv", type=str, required=True, help="CSV file of one SCADA sample")
    parser.add_argument("--tabnet_ckpt", type=str, required=True, help="Path to pretrained TabNet .pth")
    parser.add_argument("--cap_ckpt_dir", type=str, required=True, help="Dir containing args.json & *.pt")
    parser.add_argument("--time_range", type=str, default="", help="Optional time range string like '12:00~13:00'")
    parser.add_argument("--beam_size", type=int, default=5)
    args = parser.parse_args()

    # ---- 事前学習済み CaptionModel をロード ----
    args_json = os.path.join(args.cap_ckpt_dir, "args.json")
    with open(args_json) as f:
        cfg = json.load(f)

    # 最新の .pt を探す
    pt_files = [p for p in os.listdir(args.cap_ckpt_dir) if p.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError("No .pt in caption ckpt dir")
    pt_path = os.path.join(args.cap_ckpt_dir, sorted(pt_files)[-1])


    cap_model = build_caption_model(
        gpt_variant=cfg["rinna_gpt_name"],
        prefix_length=cfg["prefix_length"],
        prefix_dim=cfg["prefix_dim"],
        mapping_type=cfg["mapping_type"],
        num_layers=cfg["num_layers"],
        only_prefix=cfg["only_prefix"],
        pretrained_path=pt_path,
    ).to(DEVICE)

    tokenizer = T5Tokenizer.from_pretrained(
        "rinna/japanese-gpt2-medium" if cfg["rinna_gpt_name"] == "gpt_medium" else "rinna/japanese-gpt-1b"
    )
    tokenizer.pad_token = tokenizer.eos_token  # GPT系はpad_tokenが未定義なので明示

    # ---- TabNet Encoder ----
    tabnet_encoder = load_tabnet_encoder(args.tabnet_ckpt)

    # ---- Predictor ----
    predictor = Predictor(cap_model, tokenizer, tabnet_encoder, device=DEVICE)
    caption = predictor.caption(
        csv_path=args.scada_csv,
        time_range=args.time_range,
        beam_size=args.beam_size,
    )
    print("Generated caption:", caption)


if __name__ == "__main__":
    main()
