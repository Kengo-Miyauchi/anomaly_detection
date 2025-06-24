# -*- coding: utf-8 -*-
"""
inference.py - SCADA ベクトル → GPT2 で説明文を生成
=================================================
使い方例
--------
python inference.py \
    --scada_csv ./sample/001.csv \
    --tabnet_ckpt ./mnt/iot-qnap5/model/haenkaze/tabnet-pretrain-out2023-40dim/pretrained.pth \
    --cap_ckpt_dir ./ckpt/scada-run-ep10-bs8-lr2e-05 \
    --attributes_csv ./attributes.csv \
    --time_range "12:00~13:00" \
    --beam_size 5
"""
import argparse
import numpy as np
import torch
from transformers import T5Tokenizer
import os
import json
import pandas as pd

from model import CaptionModel, build_caption_model   # ← 新しい model.py 由来
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# 0. TabNet Encoder をロードするユーティリティ
# ----------------------------------------------------------------------
def load_tabnet_encoder(ckpt_path: str):
    """
    checkpoint に保存された `unsupervised_model` から encoder を取り出す想定。
    """
    full_model = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
        series = df.values  # [T, D]

        chunk_embeddings = []

        for j in range(6):  # 10分ずつ6分割
            start = j * (series.shape[0] // 6)
            end = (j + 1) * (series.shape[0] // 6)
            chunk = series[start:end]

            dataloader = create_dataloader(chunk, batch_size=128, need_shuffle=False)
            features = []

            for batch in dataloader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    step_outputs, _ = self.encoder(batch)
                encoder_out = sum(step for step in step_outputs).cpu()
                features.append(encoder_out)

            if len(features) == 0:
                continue

            chunk_feature = torch.cat(features).mean(dim=0)  # [40]
            chunk_embeddings.append(chunk_feature)

        final_embedding = torch.cat(chunk_embeddings).unsqueeze(0)  # [1, 240]
        return final_embedding.to(self.device)


    # --------------------------------------------------------------
    # 1-2. パブリック API
    # --------------------------------------------------------------
    def caption(self, csv_path: str, time_range: str = None, beam_size: int = 5, max_len: int = 64,
                temperature: float = 1.0, no_repeat_ngram_size: int = 3, prompt: str = None):
        prefix_vec = self._encode_scada(csv_path)                          # [1,prefix_dim]

        if prompt is not None:
            prompt_text = (time_range + "のデータに基づいて: " if time_range else "") + prompt
        else:
            prompt_text = (time_range + "のデータに基づいて: " if time_range else "")

        with torch.no_grad():
            prefix_embed = self.cap_model.prefix_mapper(prefix_vec)        # [1,P,E]
            prefix_embed = prefix_embed.view(1, self.cap_model.prefix_length, -1)

        captions = self._generate_beam(
            embed=prefix_embed,
            beam_size=beam_size,
            prompt=prompt_text if prompt_text else None,
            entry_length=max_len,
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return captions  # 最高スコア 1 件を返す


    # --------------------------------------------------------------
    # 1-3. ビームサーチ (ClipCap 実装を踏襲)
    # --------------------------------------------------------------
    def _generate_beam(self, embed, beam_size, prompt=None, entry_length=64, temperature=1.0, no_repeat_ngram_size=3):
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

            dummy_prefix_tokens = torch.zeros(1, self.cap_model.prefix_length, dtype=torch.long, device=device)
            tokens = torch.cat((dummy_prefix_tokens, prompt_tokens), dim=1)  # [1, P+T]
        else:
            generated = embed  # [1, P, E]
            tokens = torch.zeros(1, self.cap_model.prefix_length, dtype=torch.long, device=device)  # [1, P]

        generated = generated.expand(beam_size, *generated.shape[1:])
        tokens = tokens.expand(beam_size, *tokens.shape[1:])

        for _ in range(entry_length):
            outputs = self.cap_model.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()  # shape: [beam_size, vocab_size]

            # 🔽 no_repeat_ngram_size 対応
            if no_repeat_ngram_size is not None and tokens.size(1) >= no_repeat_ngram_size:
                for beam_idx in range(beam_size):
                    prev_tokens = tokens[beam_idx].tolist()
                    ngram_dict = {}
                    for i in range(len(prev_tokens) - no_repeat_ngram_size + 1):
                        prefix = tuple(prev_tokens[i : i + no_repeat_ngram_size - 1])
                        next_token = prev_tokens[i + no_repeat_ngram_size - 1]
                        ngram_dict.setdefault(prefix, set()).add(next_token)

                    current_prefix = tuple(prev_tokens[-(no_repeat_ngram_size - 1):])
                    blocked = ngram_dict.get(current_prefix, set())
                    for token_id in blocked:
                        logits[beam_idx, token_id] = -float("inf")

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

                next_tok_squeezed = next_tok.squeeze(1) if next_tok.dim() > 1 else next_tok
                stop_flags = next_tok_squeezed.eq(stop_idx)
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
    parser.add_argument("--attributes_csv", type=str, required=True, help="CSV file listing attribute names")
    parser.add_argument("--time_range", type=str, default="", help="Optional time range string like '12:00~13:00'")
    parser.add_argument("--beam_size", type=int, default=5)
    args = parser.parse_args()

    # --- 属性CSVから属性語句を読み込み ---
    df_attr = pd.read_csv(args.attributes_csv)
    # ここは属性名が1列目にある想定。列名が違うなら df_attr['列名'] に修正してください
    attr_list = df_attr.iloc[:, 0].dropna().unique().tolist()

    # プロンプト文字列作成
    prompt = "以下の語句のいずれかを必ず含んで、異常の説明文を生成してください：\n"
    for attr in attr_list:
        prompt += f"・{attr}\n"
    prompt += "\n→ "

    # 事前学習済み CaptionModel をロード
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

    tabnet_encoder = load_tabnet_encoder(args.tabnet_ckpt)

    predictor = Predictor(cap_model, tokenizer, tabnet_encoder, device=DEVICE)
    captions = predictor.caption(
        csv_path=args.scada_csv,
        time_range=args.time_range,
        beam_size=args.beam_size,
        prompt=prompt,
        temperature=0.5,
        max_len=16,
        no_repeat_ngram_size=4,
    )
    for i, c in enumerate(captions):
        print(f"Generated caption {i}: {c}")


if __name__ == "__main__":
    main()
