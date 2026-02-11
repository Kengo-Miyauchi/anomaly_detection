import torch
import pandas as pd
import numpy as np
import os
import gc
import pickle
from sklearn.mixture import GaussianMixture

from util_module.data_to_plot import plot_by_date
from util_module.calc_scores import calc_scores, calc_scores_with_windowing, calc_scores_windowing_once
from util_module.end_info import show_info
from util_module.windowing import make_windows_3d, window_timestamps, fit_standardizer, apply_standardizer
from util_module.extract_features_mtfa import data_to_MTFAFeatures
from util_module.TFMAE.build_exec_model_mtfa import ExecModelMTFA  # フォルダ名は実際に合わせて統一
from util_module.tabnet.set_config import set_config_file          # TabNetと同じ流儀で使う
from util_module import SCADA_utils

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# download data（tabnet-gmmと同じ）
# -----------------------------
stampcol = "DateTime"
frequency = '10M'
sequence_length = 10
stride = 1                   # MTFA側で追加：window stride
threshold_line = 100 - 1
dataset_name = "haenkaze"
base_path = "/mnt/work-qnap/miyauchi"

if (frequency == '1S') or ('sampled' in frequency):
    parquet_file = f'{base_path}/data/{dataset_name}/2023_{frequency}_data.parquet'
else:
    parquet_file = f'{base_path}/data/{dataset_name}/2023_{frequency}_avg_data.parquet'

print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
data = SCADA_utils.arrange_data(data, stampcol)
print("Finished loading data.")

# preprocessing
train_range = ('2023-04-01','2023-04-30')
test_range  = ('2023-04-01','2023-09-30')

train = SCADA_utils.extract_specific_terms(data, train_range[0], train_range[1], stampcol)
test  = SCADA_utils.extract_specific_terms(data, test_range[0],  test_range[1],  stampcol)
timestamp = test[stampcol]
del data

X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
X_test  = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train, test
gc.collect()

# -----------------------------
# MTFAの前処理：標準化 → window化 → timestamp整合
# -----------------------------
mu, std = fit_standardizer(X_train)
X_train = apply_standardizer(X_train, mu, std)
X_test  = apply_standardizer(X_test,  mu, std)

L = sequence_length
W_train = make_windows_3d(X_train, L, stride)  # [Ntr, L, C]
W_test  = make_windows_3d(X_test,  L, stride)  # [Nte, L, C]

timestamp_w = window_timestamps(
    timestamp.values if hasattr(timestamp, "values") else timestamp,
    L, stride, mode="median"
)
timestamp_w = timestamp_w[:len(W_test)]

del X_train, X_test
gc.collect()

# -----------------------------
# set execute model（tabnet-gmmと同じ流儀）
# -----------------------------
model_name = f"mtfa-{sequence_length}seq"
path_to_pretrained = f"model/{dataset_name}/{model_name}-40dim-{sequence_length}seq"

config = set_config_file()  # ← TabNetと同じ。返すyamlにMTFA用項目も入れる
exec_model = ExecModelMTFA(
    device=device,
    config=config,
    dataset_name=dataset_name,
    model_name=model_name,
    path_to_pretrained=path_to_pretrained,
    # MTFA側で必要な最小情報：入力次元と窓長
    c_in=W_train.shape[-1],
    win_size=L,
    seq_size=L,
)
out_dir = exec_model.out_dir  # TabNetと同じ書き味

# -----------------------------
# feature extraction
# -----------------------------
print("Start feature extraction")
feature_train = data_to_MTFAFeatures(exec_model, W_train, batch_size=256)
feature_test  = data_to_MTFAFeatures(exec_model, W_test,  batch_size=256)
print("Finish feature extraction")

del W_train, W_test
gc.collect()

# -----------------------------
# GMM training（tabnet-gmmと同じ書き味）
# -----------------------------
event_files = [f"{base_path}/data/{dataset_name}/events.csv"]
event_files.append(f"{base_path}/data/{dataset_name}/event_range.csv")

score_file = f"result/{dataset_name}/{model_name}/{frequency}scores.csv"
os.makedirs(os.path.dirname(score_file), exist_ok=True)

n_components = 10
isTrain = False

path_to_gmm = f"{base_path}/model/{dataset_name}/{model_name}"
os.makedirs(path_to_gmm, exist_ok=True)
gmm_model = f"{path_to_gmm}/gmm_{frequency}.pkl"

if os.path.exists(gmm_model) and not isTrain:
    print(f"Load from {gmm_model}")
    with open(gmm_model, "rb") as file:
        gmm = pickle.load(file)
else:
    print("GMM Training")
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=exec_model.covariance_type,  # TabNetと同じ
        random_state=42,
        n_init=10,
        max_iter=25
    )
    gmm.fit(feature_train)
    print("Finish GMM Training")
    with open(gmm_model, "wb") as file:
        pickle.dump(gmm, file)
    print(f"Model saved as {gmm_model}")

log_likelihood = -gmm.score_samples(feature_train)
threshold = np.percentile(log_likelihood, threshold_line)
del feature_train

anomaly_score = -gmm.score_samples(feature_test)
del feature_test

timestamp_plot = timestamp_w[:len(anomaly_score)]
df = pd.DataFrame({"DATETIME": timestamp_plot, "AnomalyScore": anomaly_score})

if not pd.api.types.is_datetime64_any_dtype(df['DATETIME']):
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], errors="coerce")

# 外れ値除去
num_remove = 15
remove_indices = df.nlargest(num_remove, 'AnomalyScore').index
df = df.drop(remove_indices)

# plot
log_plot = False
img_path = f"{out_dir}/{frequency}_log.png" if log_plot else f"{out_dir}/{frequency}.png"
plot_by_date(log_plot, anomaly_score, timestamp_plot, train_range, threshold, img_path, event_files)
print(f"Plot result: {img_path}")

show_info(out_dir, exec_model)
calc_scores_windowing_once(df, threshold, threshold_line, frequency, [event_files[0]], score_file, before_max=7, window='10min')
#calc_scores_with_windowing(df, threshold, threshold_line, frequency, [event_files[0]], score_file, before_max=7, window='10min')
print(f"Scores saved to {score_file}")