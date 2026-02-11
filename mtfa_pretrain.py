import torch
import pandas as pd
import numpy as np
import os
import gc

from torch.utils.data import Dataset, DataLoader

from util_module.windowing import make_windows_3d, fit_standardizer, apply_standardizer
from util_module.TFMAE.build_exec_model_mtfa import ExecModelTFMAE  # あなたが置いたファイル名に合わせて変更
from util_module.tabnet.set_config import set_config_file
from util_module import SCADA_utils


# =========================
# DataLoader用 Dataset
# =========================
class WindowDataset(Dataset):
    def __init__(self, windows_3d: np.ndarray):
        self.x = windows_3d.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


def create_window_dataloader(windows_3d: np.ndarray, batch_size: int, shuffle: bool):
    return DataLoader(
        WindowDataset(windows_3d),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )


# =========================
# main
# =========================
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

dataset_name = "haenkaze"
base_path = "/mnt/work-qnap/miyauchi"

# ---- 事前学習の窓長（TabNetのsequence_lengthに相当）----
sequence_length = 10
stride = 1

# preprocessing
target_file = f"{base_path}/data/{dataset_name}/fixed_data/2023_fixed.parquet"
parquet_file = f"{base_path}/data/{dataset_name}/fixed_data/2024_fixed.parquet"

print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
stampcol = "DateTime"

train_range = ("2024-04-01", "2024-08-31")
valid_range = ("2024-09-01", "2024-09-30")

train = SCADA_utils.extract_specific_terms(data, train_range[0], train_range[1], stampcol)
valid = SCADA_utils.extract_specific_terms(data, valid_range[0], valid_range[1], stampcol)

X_train = train.drop(columns=["DateTime", " 日付ﾌｫｰﾏｯﾄ 時分秒"]).values
X_valid = valid.drop(columns=["DateTime", " 日付ﾌｫｰﾏｯﾄ 時分秒"]).values

del train, valid, data
gc.collect()
print("Finished loading main train/valid.")

# ---- 他年度・他ファイルを train に結合（TabNet版と同じ）----
data_dir = f"{base_path}/data/{dataset_name}/fixed_data"
file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
for file_path in file_list:
    if (file_path != parquet_file) and (file_path != target_file):
        print(f"Loading data from {file_path}...")
        df = pd.read_parquet(file_path)
        df.fillna(method="ffill", inplace=True)
        arr = df.drop(columns=["DateTime", " 日付ﾌｫｰﾏｯﾄ 時分秒"]).values

        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"[Warn] {file_path}: isNAN={np.isnan(arr).any()} isINF={np.isinf(arr).any()}")

        X_train = np.concatenate([X_train, arr], axis=0)

        del df, arr
        gc.collect()
        print("Finished loading.")

# ---- 標準化：trainでfit→validも同じ変換（強く推奨）----
mu, std = fit_standardizer(X_train)
X_train = apply_standardizer(X_train, mu, std)
X_valid = apply_standardizer(X_valid, mu, std)

# ---- window化（重要：ファイル結合後に作ると境界跨ぎ窓が混ざるのが嫌なら、
#     本当は「各ファイルごとにwindow化→concat」推奨。まずは既存構造を踏襲して全体でwindow化）----
W_train = make_windows_3d(X_train, L=sequence_length, stride=stride)  # [Ntr, L, C]
W_valid = make_windows_3d(X_valid, L=sequence_length, stride=stride)  # [Nva, L, C]

del X_train, X_valid
gc.collect()

print(f"W_train: {W_train.shape}, W_valid: {W_valid.shape}")

# ---- DataLoader作成 ----
batch_size_pre = 32
train_loader = create_window_dataloader(W_train, batch_size=batch_size_pre, shuffle=True)
valid_loader = create_window_dataloader(W_valid, batch_size=batch_size_pre, shuffle=False)

# ---- ExecModelTFMAE（solver.pyのlossで学習する版）----
model_name = "mtfa-pretrain"
config = set_config_file()  # TabNetと同様、yamlパス（中身にMTFA設定が入っているのが理想）

exec_model = ExecModelTFMAE(
    device=device,
    config=config,
    dataset_name=dataset_name,
    model_name=model_name,
    train_loader=train_loader,
    valid_loader=valid_loader,
    refit=False,
    proj_dim=40,
    # MTFAに必要な情報（yamlに書くならここは省略可）
    c_in=W_train.shape[-1],
    sequence_length=sequence_length,
    seq_size=sequence_length,
    # optimizer/epoch（yamlに書くなら省略可）
    optimizer_params={"lr": 1e-3},
    max_epochs=50,
    covariance_type="diag",
)

print(f"finish model build: {exec_model.path_to_pretrained}")

# 学習曲線保存
exec_model.train_curve()
