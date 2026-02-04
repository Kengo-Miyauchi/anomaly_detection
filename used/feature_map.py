import torch
import pandas as pd
import numpy as np
import gc
from util_module import SCADA_utils

# set device
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download data
stampcol = "DateTime"
frequency = '1S'
sequence_length = 10
threshold_line = 100 - 1
dataset_name = "haenkaze"
data_path = "/mnt/work-qnap/miyauchi"
if(frequency=='1S' or 'sampled' in frequency):
    parquet_file = f'{data_path}/data/{dataset_name}/2023_'+frequency+'_data.parquet'
else:
    parquet_file = f'{data_path}/data/{dataset_name}/2023_'+frequency+'_avg_data.parquet'
print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
#data = utils.fix_data(data)
data = SCADA_utils.arrange_data(data,stampcol)
print("Finished loading data.")
feature_names = data.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).columns

# preprocessing
#data[stampcol] = pd.to_datetime(data[stampcol])
normal_range = ('2023-04-01','2023-04-30')
anomaly_range = ('2023-07-01','2023-07-31')
normal = SCADA_utils.extract_specific_terms(data,normal_range[0],normal_range[1],stampcol)
anomaly = SCADA_utils.extract_specific_terms(data,anomaly_range[0],anomaly_range[1],stampcol)
del data

X_normal = normal.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del normal
X_anomaly = anomaly.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del anomaly

# memory management
gc.collect()

path_to_pretrained = "model/haenkaze/tabnet-pretrain-out2023-40dim"
#path_to_pretrained = f"model/haenkaze/tabnet-self-attn-40dim-10seq"
unsupervised_model = torch.load(path_to_pretrained+"/pretrained.pth",weights_only=False)

print("Calculating feature importance...")
M_explain_normal, masks_normal = unsupervised_model.explain(X_normal)
print("Finished calculating feature importance.")

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

plt.figure(figsize=(12, 8))
sns.heatmap(M_explain_normal, cmap="viridis",yticklabels=False)
plt.title("TabNetの特徴量マスク (M_explain)", fontsize=16)
plt.xlabel("特徴量", fontsize=12)
plt.ylabel("サンプル", fontsize=12)
plt.tight_layout()
plt.show()
img_path = "feature_map_normal_full.png"
plt.savefig(img_path, dpi=300)
print(f"Normal data feature map saved: {img_path}")


M_explain_anomaly, masks_anomaly = unsupervised_model.explain(X_anomaly)
plt.figure(figsize=(12, 8))
sns.heatmap(M_explain_anomaly, cmap="viridis", yticklabels=False)
plt.title("TabNetの特徴量マスク (M_explain)", fontsize=16)
plt.xlabel("特徴量", fontsize=12)
plt.ylabel("サンプル", fontsize=12)
plt.tight_layout()
plt.show()
img_path = "feature_map/feature_map_anomaly_full.png"
plt.savefig(img_path, dpi=300)
print(f"Anomaly data feature map saved: {img_path}")