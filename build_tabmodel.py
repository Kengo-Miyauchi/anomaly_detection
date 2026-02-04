import torch
import pandas as pd
import numpy as np
import os
import gc
from util_module.tabnet.build_exec_model import ExecModel
from util_module.tabnet.set_config import set_config_file
from util_module import SCADA_utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download data
dataset_name = "eco_power"
base_dir = "/mnt/iot-qnap5/miyauchi/data/eco_power"
parquet_file = f'data/{dataset_name}/2020_fixed.parquet'
#parquet_file = f'data/{dataset_name}/2024_10s_avg_data.parquet'

""" print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
print("Finished loading data.") """


ID_list = ['T3']

for ID in ID_list:
    print(f"\n=== Processing {ID} ===")
    data = pd.DataFrame()

    # --- 全年度結合 ---
    for year in range(18, 25):
        data_path = f'{base_dir}/Stats_全体_{ID}_{year}0401_000000-{year+1}0331_235959.csv'
        try:
            df = pd.read_csv(data_path, encoding='shift-jis', skipfooter=1, engine='python')
            if df.shape[1] > 1 :
                data = pd.concat([data, df], ignore_index=True)
        except Exception as e:
            print(f"⚠ Error reading {data_path}: {e}")
            continue

    if data.empty:
        print(f"⚠ {ID}: データなし")
        continue

# preprocessing
stampcol = "日時"
data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2018-04-01','2022-03-31')
valid_range = ('2022-04-01','2022-08-31')

train = SCADA_utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
valid = SCADA_utils.extract_specific_terms(data,valid_range[0],valid_range[1],stampcol)

X_train = train.drop(columns=['日時']).values
del train

X_valid = valid.drop(columns=['日時']).values
del valid


# memory management
gc.collect()

# scaling data
""" from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid) """

# set execute model
model_name = "tabnet-eco_power"
config = set_config_file()
exec_model = ExecModel(device,config,dataset_name,model_name,X_train,X_valid,refit=True)
out_dir = exec_model.out_dir

print(f"finish model build: {exec_model.path_to_pretrained}")