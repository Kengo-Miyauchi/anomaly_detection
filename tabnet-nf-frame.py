import torch
import pandas as pd
import numpy as np
import os
import gc
from util_module.create_dataloader import create_dataloader
from util_module.data_to_plot import plot_by_date
from util_module.end_info import show_info
from util_module.tabnet.tabnet_feature import data_to_TabNetFeatures
from util_module.tabnet.build_exec_model import ExecModel
from util_module.tabnet.set_config import set_config_file
from util_module import SCADA_utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download dataset
dataset_name = "haenkaze"
parquet_file = f'data/{dataset_name}/2023_fixed.parquet'

print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
print("Finished loading data.")

#import pdb; pdb.set_trace()
# preprocessing
stampcol = "DateTime"
#data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2023-04-01','2023-04-30')
test_range = ('2023-04-01','2023-09-30')
train = SCADA_utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
test = SCADA_utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
timestamp = test[stampcol]

X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train
gc.collect()
#import pdb; pdb.set_trace()
X_test = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del test
gc.collect()



# set execute model
model_name = "tabnet-NF"
config = set_config_file()
exec_model = ExecModel(device,config,dataset_name,model_name,X_train)
out_dir = exec_model.out_dir

# convert data to tabnet encoder features
print("Start Feature Extraction")
feature_train = data_to_TabNetFeatures(exec_model,X_train)
feature_test = data_to_TabNetFeatures(exec_model,X_test)
print("Finish Feature Extraction")


# Normalizing Flow
from modeling.Flow_based.RealNVP import RealNVP
nf_layers = 32
dim = feature_train.shape[1]
normal_dist = torch.distributions.multivariate_normal.MultivariateNormal(
    loc=torch.zeros(dim).to(device),
    covariance_matrix=torch.eye(dim).to(device)
)

# RealNVPモデルのインスタンス化
#import pdb; pdb.set_trace()
nf_model = RealNVP(dims=[dim],cfg = {"layers": nf_layers})
nf_model.to(device)


from torch.utils.data import DataLoader, TensorDataset
def concat_frames(data, num_frames):
    """
    二次元配列を行方向に指定したフレーム数ごとに結合する関数

    Args:
        data (numpy.ndarray): 入力データ（例: shape = [T, D]）
        num_frames (int): 結合するフレーム数

    Returns:
        numpy.ndarray: 結合後のデータ
    """
    T, D = data.shape
    output = []
    for i in range(T - num_frames + 1):
        # num_frames分のデータを結合
        concatenated = data[i:i + num_frames].flatten()
        output.append(concatenated)
    return np.array(output)

def create_loader(data,batch_size,need_shuffle):
    # フレーム数を指定して結合
    num_frames = 5
    concatenated_data = concat_frames(data, num_frames)
    print("Concatenated data shape:", concatenated_data.shape)  # shape = [98, 15]
    # テンソルに変換
    tensor_data = torch.tensor(concatenated_data, dtype=torch.float32)
    # DataLoaderの作成
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=need_shuffle)
    return dataloader

train_dataloader = create_loader(feature_train, batch_size=256, need_shuffle=True)
test_dataloader = create_loader(feature_test, batch_size=256, need_shuffle=False)


# Optimizerの設定
optimizer = torch.optim.Adam(nf_model.parameters(), lr=1e-3)

# 学習ループ
print("NF Training...")
for epoch in range(50):
    nf_model.train()
    epoch_loss = 0.0
    for batch in train_dataloader:
        batch = batch.to(device)
        z, log_df_dz = nf_model.forward(batch)
        loss = torch.mean(-normal_dist.log_prob(z) - log_df_dz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        del batch, z, log_df_dz
    epoch_loss /= len(train_dataloader)
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}")
gc.collect()
print("Finish NF Training")

print("Calculate Train Score")
# 正常データの異常スコアを計算
normal_scores = []
nf_model.eval()
with torch.no_grad():
    for batch in train_dataloader:
        batch = batch.to(device)
        z, log_df_dz = nf_model.forward(batch)
        log_likelihood = normal_dist.log_prob(z) + log_df_dz
        normal_scores.extend(-log_likelihood.cpu().numpy())
        del batch, z, log_df_dz
threshold = np.percentile(normal_scores, 99)
gc.collect()

print("Calculate Test Score")
# テストデータの異常スコアを計算
test_scores = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = batch.to(device)
        z, log_df_dz = nf_model.forward(batch)
        log_likelihood = normal_dist.log_prob(z) + log_df_dz
        test_scores.extend(-log_likelihood.cpu().numpy())
        del batch, z, log_df_dz
gc.collect()

plot_by_date(exec_model.log_plot,test_scores,timestamp,train_range,threshold,out_dir)
show_info(out_dir,exec_model)