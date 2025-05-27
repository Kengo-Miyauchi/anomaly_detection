import torch
import pandas as pd
import numpy as np
import os
import gc
from util_module.tabnet.build_exec_model import ExecModel
from util_module.tabnet.set_config import set_config_file
from util_module import SCADA_utils

# set device
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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

# set execute model
#model_name = "tabnet-pretrain-tout2023"
model_name = "tabnet-self-attn"
config = set_config_file()
exec_model = ExecModel(device,config,dataset_name,model_name,X_train,X_valid)
out_dir = exec_model.out_dir

print(f"finish model build: {exec_model.path_to_pretrained}")

exec_model.train_curve()