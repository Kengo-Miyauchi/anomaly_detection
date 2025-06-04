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

M_explain_normal, masks_normal = unsupervised_model.explain(X_normal)
M_explain_anomaly, masks_anomaly = unsupervised_model.explain(X_anomaly)

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

# 平均をとって全体の重要度に変換（全体的な可視化）
mean_importance_normal = np.mean(M_explain_normal, axis=0)
mean_importance_anomaly = np.mean(M_explain_anomaly, axis=0)

importance_df_normal = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_importance_normal
})
# 重要度の降順でソート
importance_df_normal = importance_df_normal.sort_values(by='importance', ascending=False)
top_k = 10
top_features_normal = importance_df_normal.head(top_k)
sns.barplot(data=top_features_normal, x='importance', y='feature', palette='viridis')
plt.title("TabNet Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
plt.savefig("feature_map/feature_importance_bar.png")
print("Feature Importance Bar Plot saved: feature_map/feature_importance_bar.png")

importance_df_anomaly = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_importance_anomaly
})
# 重要度の降順でソート
importance_df_anomaly = importance_df_anomaly.sort_values(by='importance', ascending=False)
top_k = 10
top_features_anomaly = importance_df_anomaly.head(top_k)
sns.barplot(data=top_features_anomaly, x='importance', y='feature', palette='viridis')
plt.title("TabNet Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
plt.savefig("feature_map/feature_importance_bar_anomaly.png")
print("Feature Importance Bar Plot (Anomaly) saved: feature_map/feature_importance_bar_anomaly.png")
