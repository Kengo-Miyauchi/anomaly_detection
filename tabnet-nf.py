import torch
import pandas as pd
import numpy as np
import os
import gc
from util_module.create_dataloader import create_dataloader
from util_module.data_to_plot import plot_by_date,calc_scores
from util_module.end_info import show_info
from util_module.tabnet_feature import data_to_TabNetFeatures
from util_module.build_exec_model import ExecModel
from util_module.set_config import set_config_file
from wind_module import utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download dataset
stampcol = "DateTime"
frequency = '10S'
threshold_line = 100 - 0.001
max_epoch = 20
isTrain = False

dataset_name = "haenkaze"
if(frequency=='1S' or 'sampled' in frequency):
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_data.parquet'
else:
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_avg_data.parquet'

print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
data = utils.arrange_data(data,stampcol)
print("Finished loading data.")

#import pdb; pdb.set_trace()
# preprocessing
#data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2023-04-01','2023-04-30')
test_range = ('2023-04-01','2023-09-30')
train = utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
test = utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
timestamp = test[stampcol]

X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train
gc.collect()
#import pdb; pdb.set_trace()
X_test = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del test
gc.collect()



# set execute model
model_name = "tabnet-nf"
config = set_config_file()
exec_model = ExecModel(device,config,dataset_name,model_name,X_train)
out_dir = f"{exec_model.out_dir}/{frequency}"
os.makedirs(out_dir, exist_ok=True)

# convert data to tabnet encoder features
print("Start feature extraction")
feature_train = data_to_TabNetFeatures(exec_model,X_train)
feature_test = data_to_TabNetFeatures(exec_model,X_test)
print("Finish feature extraction")


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

# DataLoaderの作成
train_dataloader = create_dataloader(feature_train.numpy(), batch_size=256, need_shuffle=True)
test_dataloader = create_dataloader(feature_test.numpy(), batch_size=256, need_shuffle=False)
del feature_train,feature_test

# Optimizerの設定
optimizer = torch.optim.Adam(nf_model.parameters(), lr=1e-3)

trained_nf = f"model/haenkaze/{model_name}/nf_{frequency}.pth"
if os.path.exists(trained_nf) and (not isTrain):
    print(f"Load from {trained_nf}")
    with open(trained_nf, "rb") as file:
        nf_model.load_state_dict(torch.load(trained_nf))
        nf_model.eval()
else:
    # 学習ループ
    print("NF Training...")
    epoch_losses = []
    for epoch in range(max_epoch):
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
        epoch_losses.append(epoch_loss)
    gc.collect()
    print("Finish NF Training")
    with open(trained_nf, "wb") as file:
        torch.save(nf_model.state_dict(), trained_nf)
    print("Model saved as nf_model.pth")  
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(epoch_losses)), epoch_losses, label="Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    img_path = f"{out_dir}/train_curve.png"
    plt.savefig(img_path)
    print(img_path)


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
threshold = np.percentile(normal_scores, threshold_line)
del train_dataloader
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
del test_dataloader
gc.collect()


df = pd.DataFrame({"DATETIME": timestamp, "AnomalyScore": test_scores})
if(not(pd.api.types.is_datetime64_any_dtype(df['DATETIME']))):
    df['DATETIME'] = pd.to_datetime(df['DATETIME'],format='%d/%m/%y %H')
# 外れ値の除去
num_remove = 15
# AnomalyScore列の上位num_remove個のインデックスと値を取得
remove_indices = df.nlargest(num_remove, 'AnomalyScore').index
# データフレームから削除
df = df.drop(remove_indices) 
df['Date'] = df['DATETIME'].dt.date

event_files = ["data/haenkaze/events.csv"]
event_files.append("data/haenkaze/event_range.csv")
score_file = f"{out_dir}/scores.csv"
log_plot = False
if log_plot:img_path = f"{out_dir}/{frequency}_log.png"
else:img_path = f"{out_dir}/{frequency}.png"
plot_by_date(log_plot,test_scores,timestamp,train_range,threshold,img_path,event_files)
print(f"Threshold line: {threshold_line}%")
calc_scores(df, threshold, threshold_line, frequency, [event_files[0]], score_file, before_max=7)
print(f"result: {img_path}")
show_info(out_dir,exec_model)
print(f"Score result: {score_file}")
