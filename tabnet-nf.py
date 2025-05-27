import torch
import pandas as pd
import numpy as np
import os
import gc
import time
from matplotlib import pyplot as plt
from util_module.create_dataloader import create_dataloader
from util_module.data_to_plot import plot_by_date,calc_scores
from util_module.end_info import show_info
from util_module.tabnet.extract_features import data_to_TabNetFeatures
from util_module.tabnet.build_exec_model import ExecModel
from util_module.tabnet.set_config import set_config_file
from util_module import SCADA_utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download dataset
stampcol = "DateTime"
frequency = '1S'
threshold_line = 100 - 0.1
max_epoch = 50
nf_train = False
train_continue = False

base_path = "/mnt/work-qnap/miyauchi"
model_name = "tabnet-nf"
dataset_name = "haenkaze"
feature_path = f"{base_path}/data/{dataset_name}/features/{frequency}"
anomaly_score_path = f"{base_path}/data/{dataset_name}/anomaly_score/{model_name}"


if(frequency=='1S' or 'sampled' in frequency):
        parquet_file = f'{base_path}/data/{dataset_name}/2023_'+frequency+'_data.parquet'
else:
    parquet_file = f'{base_path}/data/{dataset_name}/2023_'+frequency+'_avg_data.parquet'

print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
data = SCADA_utils.arrange_data(data,stampcol)
print("Finish loading data.")

# preprocessing
train_range = ('2023-04-01','2023-04-30')
test_range = ('2023-04-01','2023-09-30')
train = SCADA_utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
test = SCADA_utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
timestamp = test[stampcol]
X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train
gc.collect()
X_test = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del test
gc.collect()
# set execute model
config = set_config_file()
exec_model = ExecModel(device,config,dataset_name,model_name,X_train)
out_dir = f"{exec_model.out_dir}/{frequency}"
os.makedirs(out_dir, exist_ok=True)

# load features
if os.path.exists(f"{feature_path}/feature_train.npy"):
    del X_train,X_test
    print("Loading features")
    feature_train = np.load(f"{feature_path}/feature_train.npy")
    feature_test = np.load(f"{feature_path}/feature_test.npy")
    print("Finish Loading features")
else: #extract_features
    # convert data to tabnet encoder features
    print("Start feature extraction")
    feature_train = data_to_TabNetFeatures(exec_model,X_train)
    np.save(f"{feature_path}/feature_train.npy", feature_train)
    del X_train
    feature_test = data_to_TabNetFeatures(exec_model,X_test)
    np.save(f"{feature_path}/feature_test.npy", feature_test)
    del X_test
    print("Finish feature extraction")
    feature_train.numpy()
    feature_test.numpy()

# Normalizing Flow
from modeling.Flow_based.RealNVP import RealNVP
nf_layers = 32
dim = feature_train.shape[1]
normal_dist = torch.distributions.multivariate_normal.MultivariateNormal(
    loc=torch.zeros(dim).to(device),
    covariance_matrix=torch.eye(dim).to(device)
)
nf_model = RealNVP(dims=[dim],cfg = {"layers": nf_layers})
nf_model.to(device)
train_dataloader = create_dataloader(feature_train, batch_size=256, need_shuffle=True)
test_dataloader = create_dataloader(feature_test, batch_size=256, need_shuffle=False)
del feature_train,feature_test

optimizer = torch.optim.Adam(nf_model.parameters(), lr=1e-4)

trained_nf = f"{base_path}/model/haenkaze/{model_name}/nf_{frequency}_{max_epoch}epoch.pth"
if os.path.exists(trained_nf) and (not nf_train):
    print(f"Load from {trained_nf}")
    with open(trained_nf, "rb") as file:
        checkpoint = torch.load(trained_nf)
        nf_model.load_state_dict(torch.load(trained_nf))
    if train_continue:
        nf_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.Adam(nf_model.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        num_epochs = 10
        total_epochs = start_epoch + num_epochs
        for epoch in range(max_epoch):
            start_time = time.time()
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
            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Epoch [{epoch+1}/{max_epoch}] | Loss = {epoch_loss:.4f} | {int(hours)}h {int(minutes)}m {int(seconds)}s")
            #epoch_losses.append(epoch_loss)
    else: nf_model.eval()
else:
    print("NF Training...")
    epoch_losses = []
    for epoch in range(max_epoch):
        start_time = time.time()
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
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Epoch [{epoch+1}/{max_epoch}] | Loss = {epoch_loss:.4f} | {int(hours)}h {int(minutes)}m {int(seconds)}s")
        epoch_losses.append(epoch_loss)
    gc.collect()
    print("Finish NF Training")
    with open(trained_nf, "wb") as file:
        torch.save(nf_model.state_dict(), trained_nf)
    print(f"Model saved as {trained_nf}")
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(epoch_losses)), epoch_losses, label="Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    train_curve = f"{out_dir}/train_curve.png"
    plt.savefig(train_curve)

if os.path.exists(f"{anomaly_score_path}/{frequency}_train.npy"):
    train_scores = np.load(f"{anomaly_score_path}/{frequency}_train.npy")
    threshold = np.percentile(train_scores, threshold_line)
    del train_dataloader
    gc.collect()
    test_scores = np.load(f"{anomaly_score_path}/{frequency}_test.npy")
    del test_dataloader
    gc.collect()
else:
    print("Start Train Score Calculation")
    # 学習区間の異常スコアを計算
    train_scores = []
    nf_model.eval()
    with torch.no_grad():
        for batch in train_dataloader:
            batch = batch.to(device)
            z, log_df_dz = nf_model.forward(batch)
            log_likelihood = normal_dist.log_prob(z) + log_df_dz
            train_scores.extend(-log_likelihood.cpu().numpy())
            del batch, z, log_df_dz
    threshold = np.percentile(train_scores, threshold_line)
    del train_dataloader
    gc.collect()
    print("Finish Train Score Calculation")
    os.makedirs(anomaly_score_path, exist_ok=True)
    np.save(f"{anomaly_score_path}/{frequency}_train.npy", train_scores)

    print("Start Test Score Calculation")
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
    print("Finish Test Score Calculation")
    np.save(f"{anomaly_score_path}/{frequency}_test.npy", test_scores)

df = pd.DataFrame({"DATETIME": timestamp, "AnomalyScore": test_scores})
if(not(pd.api.types.is_datetime64_any_dtype(df['DATETIME']))):
    df['DATETIME'] = pd.to_datetime(df['DATETIME'],format='%d/%m/%y %H')
# 外れ値の除去
num_remove = 15
remove_indices = df.nlargest(num_remove, 'AnomalyScore').index
df = df.drop(remove_indices) 
df['Date'] = df['DATETIME'].dt.date

event_files = ["data/haenkaze/events.csv"]
event_files.append("data/haenkaze/event_range.csv")
score_file = f"{out_dir}/scores.csv"
log_plot = False
if log_plot:img_path = f"{out_dir}/{frequency}_log.png"
else:img_path = f"{out_dir}/{frequency}.png"
plot_by_date(log_plot,test_scores,timestamp,train_range,threshold,img_path,event_files)
calc_scores(df, threshold, threshold_line, frequency, [event_files[0]], score_file, before_max=7)
print(f"result: {img_path}")
print(f"Score result: {score_file}")
if nf_train: print(f"Train curve: {train_curve}")

show_info(out_dir,exec_model)
