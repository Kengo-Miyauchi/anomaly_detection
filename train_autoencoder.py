import torch
import pandas as pd
import numpy as np
import os
import gc
import time
from datetime import datetime
import matplotlib.pyplot as plt
from util_module.autoencoder import AutoEncoder
from util_module import SCADA_utils
from util_module.create_dataloader import create_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
# download data
dataset_name = "haenkaze"
base_path = "/mnt/work-qnap/miyauchi"

# preprocessing
target_file = f'{base_path}/data/{dataset_name}/fixed_data/2023_fixed.parquet'
parquet_file = f'{base_path}/data/{dataset_name}/fixed_data/2024_fixed.parquet'
print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
stampcol = "DateTime"
train_range = ('2024-04-01','2024-08-31')
valid_range = ('2024-09-01','2024-09-30')
train = SCADA_utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
valid = SCADA_utils.extract_specific_terms(data,valid_range[0],valid_range[1],stampcol)
X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train,data
X_valid = valid.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del valid
gc.collect()
print("Finished loading data.")

data_dir =f'{base_path}/data/haenkaze/fixed_data'
file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
for file_path in file_list:
    if((file_path != parquet_file) and (file_path != target_file)):
        print(f"Loading data from {file_path}...")
        df = pd.read_parquet(file_path)
        df.fillna(method="ffill",inplace=True)
        data = df.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
        if(np.isnan(data).any() or np.isinf(data).any()):
            print(f"isNAN: {np.isnan(data).any()}")
            print(f"isINF: {np.isinf(data).any()}")
        X_train = np.concatenate([X_train,data])
        del df,data
        print("Finished loading data.")
        gc.collect()

# ハイパーパラメータ
input_dim = X_train.shape[1]
hidden_dim = 40
num_epochs = 100
batch_size = 1024
learning_rate = 1e-3

# ログファイル準備
log_dir = f"{base_path}/model/{dataset_name}/autoencoder/"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'log_{hidden_dim}dim.txt')

# モデルと損失関数、最適化
model = AutoEncoder(
    input_dim=input_dim,
    hidden_dim=hidden_dim
    ).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
dataloader = create_dataloader(X_train, batch_size=batch_size, need_shuffle=True)

loss_list = []
print(f"log_file: {log_file}")
# 学習ループ
with open(log_file, 'w') as f_log:
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    f_log.write(f"Start AutoEncoder training: {formatted_time}\n")
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_list.append(avg_loss)
        
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # ファイルにも保存
        f_log.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n')

# 学習済みモデルの保存
path_to_model = f"{base_path}/model/{dataset_name}/autoencoder"
torch.save(model.state_dict(), f"{path_to_model}/autoencoder_{hidden_dim}dim.pth")
print(f"Model saved: {path_to_model}/autoencoder_{hidden_dim}dim.pth")

# 学習曲線のプロット
plt.figure(figsize=(10,6))
plt.plot(range(1, num_epochs+1), loss_list, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.show()
img_path = f"{path_to_model}/train_curve.png"
plt.savefig(img_path)
print(img_path)