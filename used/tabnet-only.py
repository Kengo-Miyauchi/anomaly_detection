import torch
import pandas as pd
import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split
from util_module.data_to_plot import plot_by_date
from util_module.tabnet.build_exec_model import ExecModel
from util_module.tabnet.set_config import set_config_file
from util_module import SCADA_utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download data
dataset_name = "haenkaze"
stampcol = "DateTime"
frequency = '1S'
if(frequency=='1S' or 'sampled'in frequency):
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_data.parquet'
else:
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_avg_data.parquet'

print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
data = SCADA_utils.arrange_data(data,stampcol)
print("Finished loading data.")

# preprocessing
#data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2023-04-01','2023-04-30')
test_range = ('2023-04-01','2023-09-30')
train = SCADA_utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
test = SCADA_utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
timestamp = test[stampcol]
del data

X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train
X_test = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del test

X_train, X_valid = train_test_split(X_train, test_size=0.2, random_state=0)
# memory management
gc.collect()
X_train = X_train.astype(np.float16)
X_test = X_test.astype(np.float16)

# set execute model
model_name = "tabnet-only"
config = set_config_file()
exec_model = ExecModel(device,config,dataset_name,model_name,X_train,X_valid=X_valid,path_to_pretrained="model/haenkaze/tabnet-only-40dim")
out_dir = "result/haenkaze/tabnet-only"

print("Start Calculation")
# 学習済みモデルを使用して再構成誤差を計算
reconstructed_X_train = exec_model.unsupervised_model.predict(X_train)[0]
reconstruction_errors_train = np.mean((X_train - reconstructed_X_train) ** 2, axis=1)
threshold = np.percentile(reconstruction_errors_train, 99)
del X_train,reconstructed_X_train,reconstruction_errors_train
reconstructed_X_test = exec_model.unsupervised_model.predict(X_test)[0]
reconstruction_errors = np.mean((X_test - reconstructed_X_test) ** 2, axis=1)
del X_test, reconstructed_X_test
print("Finish Calculation")


#import pdb; pdb.set_trace()
event_files = ["data/haenkaze/events.csv","data/haenkaze/event_range.csv"]
score_file = "result/haenkaze/tabnet-only/"+frequency+"scores.csv"
img_path = out_dir + "/tabnet-only_"+frequency+".png"

plot_by_date(False,reconstruction_errors,timestamp,train_range,threshold,img_path,event_files)
print(f"result: {img_path}")
#show_info(out_dir,exec_model)
print(f"Score result: {score_file}")
""" # 再構成誤差を計算
reconstruction_errors = calc_reconstruction_errors(exec_model, X_test)

# 正常データの再構成誤差から閾値を設定（95パーセンタイルを閾値とする例）
threshold = torch.quantile(reconstruction_errors, 0.95) """