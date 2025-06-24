# -*- coding: utf-8 -*-
"""
model.py - SCADA x GPT キャプション生成モデル
============================================

* TabNet などで抽出した **SCADA 埋め込み (prefix)** を GPT-2 に渡して
  説明文を生成するモデル
* Dataset, モデル構築, 事前学習チェックポイント読込まで 1 ファイルで完結
"""
import os
import sys
import json
import pickle
from glob import glob
from enum import Enum
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, T5Tokenizer


# ---------------------------------------------------------------------
# 共通設定
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1. データセット – 事前処理済み pkl を読む
# ---------------------------------------------------------------------
class SCADADataset(Dataset):
    """
    train.pkl / valid.pkl などから
      - tokens:       [seq_len]
      - mask:         [prefix_len + prompt_len + seq_len]
      - prefix:       [prefix_dim]   (SCADA 埋め込みベクトル)
      - caption_str:  str            (デバッグ用・任意)
    を返す。
    """
    def __init__(
        self,
        data_path: str,
        prefix_length: int,
        prompt_length: int = 0,
        gpt_model_name: str = "rinna/japanese-gpt2-medium",
    ):
        self.tokenizer = T5Tokenizer.from_pretrained(gpt_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT系はpad_tokenが未定義なので明示

        self.prefix_length = prefix_length
        self.prompt_length = prompt_length

        with open(data_path, "rb") as f:
            packed = pickle.load(f)

        self.prefixes = packed["scada_embedding"]              # Tensor[N, prefix_dim]
        captions_raw = packed["captions"]                      # List[dict]
        self.captions = [c["caption"] for c in captions_raw]
        #self.time_ranges = [c["time_range"] for c in captions_raw]

        # ここでprompt_tokensを用意（例：固定プロンプトや属性語句からトークン化して保存済みのはず）
        # もしpackedに 'prompt_tokens' などなければ後で別実装が必要
        # ここでは空のprompt_tokensを返す想定（要適宜修正）
        self.prompt_token_list = []
        for c in captions_raw:
            # 例: c["prompt"] があれば tokenizer.encode して保存するなど
            self.prompt_token_list.append(torch.tensor([], dtype=torch.long))

        # --- トークン化 & 事前パディング情報 ---
        self.caption_tokens = []
        self.id2vec = []                                       # index → prefix row id
        max_len = 0
        for row in captions_raw:
            ids = torch.tensor(self.tokenizer.encode(row["caption"]), dtype=torch.long)
            self.caption_tokens.append(ids)
            self.id2vec.append(row["scada_embedding"])
            max_len = max(max_len, len(ids))

        # 動的長にしてもよいが ClipCap 互換で “平均+10σ” を採用
        all_lens = torch.tensor([len(t) for t in self.caption_tokens]).float()
        self.max_seq_len = min(int(all_lens.mean() + all_lens.std() * 10), int(all_lens.max()))

    # -------- Dataset プロトコル --------
    def __len__(self):
        return len(self.caption_tokens)

    def __getitem__(self, idx: int):
        tokens, mask = self._pad_tokens(idx)
        prefix_vec = self.prefixes[self.id2vec[idx]]
        prompt_tokens = self.prompt_token_list[idx]
        return tokens, mask, prefix_vec, prompt_tokens, self.captions[idx]

    # -------- 内部 util --------
    def _pad_tokens(self, idx: int):
        """負値を一時的に PAD 印として使い、最後に 0 に置換"""
        tokens = self.caption_tokens[idx]
        pad_len = self.max_seq_len - len(tokens)
        if pad_len > 0:
            tokens = torch.cat([tokens, torch.full((pad_len,), -1, dtype=torch.long)])
        else:
            tokens = tokens[: self.max_seq_len]

        mask = tokens.ge(0)          # valid → 1, pad → 0
        tokens = tokens.masked_fill(~mask, 0)
        mask = mask.float()
        # prefix + prompt + text の長さのマスクを作るためには
        # prompt_lengthが固定長ならその分だけ1にする必要あり（要train.pyで対応）
        # ここではmaskにprompt_length分を1で埋める処理はしない想定（train.pyで追加してください）
        mask = torch.cat([torch.ones(self.prefix_length), mask])
        return tokens, mask


# ---------------------------------------------------------------------
# 2. Prefix → GPT 埋め込みへのマッピング層
# ---------------------------------------------------------------------
class MappingType(Enum):
    MLP = "mlp"
    TRANSFORMER = "transformer"


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], act=nn.Tanh, bias=True):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class MlpBlock(nn.Module):
    """Transformer の feed-forward 相当 (2層 MLP)"""
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, act=F.relu):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = act
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadAttention(nn.Module):
    """シンプルな MHSA (Enc-Dec も可能)"""
    def __init__(self, dim_q, dim_kv, n_heads, drop=0.0, bias=False):
        super().__init__()
        self.n_heads = n_heads
        head_dim = dim_q // n_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim_q, dim_q, bias=bias)
        self.to_kv = nn.Linear(dim_kv, dim_q * 2, bias=bias)
        self.proj = nn.Linear(dim_q, dim_q)
        self.drop = nn.Dropout(drop)

    def forward(self, q, kv=None, mask=None):
        kv = kv if kv is not None else q
        B, N, C = q.shape
        _, M, _ = kv.shape

        q = self.to_q(q).reshape(B, N, self.n_heads, C // self.n_heads)
        kv = self.to_kv(kv).reshape(B, M, 2, self.n_heads, C // self.n_heads)
        k, v = kv[:, :, 0], kv[:, :, 1]

        attn = torch.einsum("bnhd,bmhd->bnmh", q, k) * self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(3)  # [B,1,N,1]
            attn = attn.masked_fill(~mask.bool(), float("-inf"))
        attn = attn.softmax(dim=2)
        out = torch.einsum("bnmh,bmhd->bnhd", attn, v).reshape(B, N, C)
        return self.proj(out), attn


class TransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, dim, n_heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MlpBlock(dim, mlp_ratio, drop)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerMapper(nn.Module):
    """
    単一ベクトル (prefix_dim) → 複数 GPT 埋め込み (prefix_len × embed_dim)
    """
    def __init__(self, dim_in, dim_embed, prefix_len, inner_len=4, n_layers=4, n_heads=8):
        super().__init__()
        self.inner_len = inner_len
        self.linear = nn.Linear(dim_in, inner_len * dim_embed)
        self.prefix_const = nn.Parameter(torch.randn(prefix_len, dim_embed))
        self.layers = nn.ModuleList(
            [TransformerLayer(dim_embed, n_heads) for _ in range(n_layers)]
        )

    def forward(self, x):
        # x: [B, dim_in] → [B, inner_len, dim_embed]
        x = self.linear(x).view(x.size(0), self.inner_len, -1)
        # 固定 prefix と連結
        prefix = self.prefix_const.unsqueeze(0).expand(x.size(0), -1, -1)  # [B, P, D]
        tok_seq = torch.cat([x, prefix], dim=1)                            # [B, L+P, D]
        for layer in self.layers:
            tok_seq = layer(tok_seq)
        return tok_seq[:, self.inner_len:]                                 # P 個だけ返す


# ---------------------------------------------------------------------
# 3. GPT2 + Prefix + Prompt モデル
# ---------------------------------------------------------------------
class CaptionModel(nn.Module):
    """
    * `prefix` (TabNet などの SCADA 埋め込みベクトル)
    * `prompt_tokens` (トークンID列, 形は [B, prompt_length])
    * `tokens` (T5Tokenizer でエンコードしたキャプション)
    を入力し、Cross-Entropy Loss を返す (train) / logits を返す (eval)。
    """

    def __init__(
        self,
        prefix_length: int,
        prompt_length: int = 0,
        prefix_dim: int = 512,
        mapping_type: str = "mlp",          # "mlp" | "transformer"
        num_layers: int = 4,                # TransformerMapper 用
        gpt_name: str = "rinna/japanese-gpt2-medium",
    ):
        super().__init__()
        self.prefix_length = prefix_length
        self.prompt_length = prompt_length
        self.gpt = AutoModelForCausalLM.from_pretrained(gpt_name)
        embed_dim = self.gpt.transformer.wte.weight.size(1)

        mapping_type = MappingType(mapping_type.lower())
        if mapping_type == MappingType.MLP:
            self.prefix_mapper = MLP(
                sizes=(
                    prefix_dim,
                    (embed_dim * prefix_length) // 2,
                    embed_dim * prefix_length,
                )
            )
        else:  # TransformerMapper
            self.prefix_mapper = TransformerMapper(
                dim_in=prefix_dim,
                dim_embed=embed_dim,
                prefix_len=prefix_length,
                n_layers=num_layers,
            )

    # ---------------- util ----------------
    @staticmethod
    def _dummy_tokens(batch, length, device):
        return torch.zeros(batch, length, dtype=torch.long, device=device)

    # --------------- forward --------------
    def forward(
        self,
        tokens: torch.Tensor,             # [B, T] (テキスト本文トークン)
        prefix: torch.Tensor,             # [B, prefix_dim]
        prompt_tokens: Optional[torch.Tensor] = None,  # [B, prompt_length]
        mask: Optional[torch.Tensor] = None,            # [B, P + prompt_length + T]
        labels: Optional[torch.Tensor] = None,
    ):
        """
        tokens : [B, T] (本文のみ)
        prefix : [B, prefix_dim]
        prompt_tokens : [B, prompt_length] or None
        mask   : [B, prefix_length + prompt_length + T]
        """
        B = tokens.size(0)
        device = tokens.device

        # prefix embedding
        prefix_emb = self.prefix_mapper(prefix).view(B, self.prefix_length, -1)    # [B, P, E]

        # prompt embedding
        if prompt_tokens is not None and prompt_tokens.size(1) > 0:
            prompt_emb = self.gpt.transformer.wte(prompt_tokens)                   # [B, prompt_length, E]
        else:
            prompt_emb = torch.empty(B, 0, prefix_emb.size(-1), device=device)     # 空テンソル

        # text embedding
        text_emb = self.gpt.transformer.wte(tokens)                               # [B, T, E]

        # full embedding: prefix + prompt + text
        full_emb = torch.cat([prefix_emb, prompt_emb, text_emb], dim=1)           # [B, P + prompt_length + T, E]

        if labels is not None:
            dummy = self._dummy_tokens(B, self.prefix_length + self.prompt_length, tokens.device)
            labels = torch.cat([dummy, tokens], dim=1)

        out = self.gpt(inputs_embeds=full_emb, attention_mask=mask, labels=labels)
        return out                                                       # loss / logits


# ---------------------------------------------------------------------
# 4. 事前学習済みモデルのロードユーティリティ
# ---------------------------------------------------------------------
def _latest_checkpoint(path_dir):
    ckpts = [p for p in glob(os.path.join(path_dir, "*.pt")) if not os.path.basename(p).startswith(".")]
    if not ckpts:
        raise FileNotFoundError(path_dir)
    return max(ckpts, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def process_pretrained_path(pretrained_path: str):
    """
    * ディレクトリを渡されたら最新 epoch の .pt を選択
    * 個別ファイルならそのまま
    * 同じ階層に args.json がある前提
    """
    if os.path.isdir(pretrained_path):
        ckpt_fpath = _latest_checkpoint(pretrained_path)
        args_fpath = os.path.join(pretrained_path, "args.json")
    else:
        ckpt_fpath = pretrained_path
        args_fpath = os.path.join(os.path.dirname(pretrained_path), "args.json")

    if not os.path.isfile(args_fpath):
        raise FileNotFoundError(args_fpath)
    return ckpt_fpath, args_fpath


def build_caption_model(
    gpt_variant: str,
    prefix_length: int,
    prefix_dim: int,
    mapping_type: str = "mlp",
    num_layers: int = 4,
    prompt_length: int = 0,
    only_prefix: bool = True,
    pretrained_path: Optional[str] = None,
):
    """
    * `only_prefix=True` の場合、GPT-2 本体は凍結し prefix_mapper のみ学習
    """
    model = CaptionModel(
        prefix_length=prefix_length,
        prompt_length=prompt_length,
        prefix_dim=prefix_dim,
        mapping_type=mapping_type,
        num_layers=num_layers,
        gpt_name=(
            "rinna/japanese-gpt2-medium" if gpt_variant == "gpt_medium" else "rinna/japanese-gpt-1b"
        ),
    )

    if only_prefix:
        for p in model.gpt.parameters():
            p.requires_grad_(False)

    if pretrained_path:
        ckpt_fpath, args_fpath = process_pretrained_path(pretrained_path)
        # --- パラメータ互換性チェック (最低限) ---
        saved_args = json.load(open(args_fpath))
        assert saved_args["prefix_length"] == prefix_length
        assert saved_args["prefix_dim"] == prefix_dim
        assert saved_args["mapping_type"] == mapping_type

        state = torch.load(ckpt_fpath, map_location="cpu")
        # DataParallel 対応
        new_state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
        model.load_state_dict(new_state)
        print(f"[INFO] loaded weights from {ckpt_fpath}")

    return model


# ---------------------------------------------------------------------
# エントリ (デバッグ用)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 簡易動作テスト
    dummy_dataset = SCADADataset(
        data_path="train.pkl",       # 例 (存在しなくてもエラー確認可)
        prefix_length=10,
        prompt_length=0,
    )
    model = build_caption_model(
        gpt_variant="gpt_medium",
        prefix_length=10,
        prefix_dim=512,
        mapping_type="mlp",
        prompt_length=0,
    ).to(DEVICE)

    try:
        sample = dummy_dataset[0]
        tokens, mask, prefix_vec, prompt_tokens, _ = sample
        tokens = tokens.unsqueeze(0).to(DEVICE)
        mask = mask.unsqueeze(0).to(DEVICE)
        prefix_vec = prefix_vec.unsqueeze(0).to(DEVICE)
        if prompt_tokens.numel() > 0:
            prompt_tokens = prompt_tokens.unsqueeze(0).to(DEVICE)
        else:
            prompt_tokens = None

        out = model(tokens=tokens, prefix=prefix_vec, prompt_tokens=prompt_tokens, mask=mask, labels=tokens)
        print("forward OK, loss =", out.loss.item())
    except Exception as e:
        print("Self-check skipped (dataset file not found).", e)
