import torch
import pandas as pd
import numpy as np
import os
import gc
import pickle
from util_module.autoencoder import AutoEncoder
from sklearn.mixture import GaussianMixture
from util_module.data_to_plot import plot_by_date,calc_scores
from util_module.tabnet.extract_features import data_to_AEfeatures
from util_module import SCADA_utils

# set device
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download data
stampcol = "DateTime"
frequency = '1S'
threshold_line = 100 - 1
dataset_name = "haenkaze"
base_path = "/mnt/work-qnap/miyauchi"
if(frequency=='1S' or 'sampled' in frequency):
    parquet_file = f'{base_path}/data/{dataset_name}/2023_'+frequency+'_data.parquet'
else:
    parquet_file = f'{base_path}/data/{dataset_name}/2023_'+frequency+'_avg_data.parquet'
print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
#data = utils.fix_data(data)
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

# memory management
gc.collect()
X_train = X_train.astype(np.float16)
X_test = X_test.astype(np.float16)


# set AutoEncoder model
model_name = "AE-gmm"
#path_to_pretrained = f"{base_path}/model/haenkaze/tabnet-pretrain-out2023-40dim"
path_to_pretrained = f"{base_path}/model/haenkaze/autoencoder/autoencoder_40dim.pth"
print(f"Load model from {path_to_pretrained}")
model = AutoEncoder(input_dim=X_train.shape[1], hidden_dim=40).to(device)
model.load_state_dict(torch.load(path_to_pretrained))
model.eval()
out_dir = "result/" + dataset_name + "/" + model_name + "/" + str(40) + "dim"
os.makedirs(out_dir, exist_ok=True)

# convert data to tabnet encoder features
print("Start feature extraction")
feature_train = data_to_AEfeatures(model,device,X_train)
feature_train = feature_train[:len(X_train)]
feature_test = data_to_AEfeatures(model,device,X_test)
feature_test = feature_test[:len(X_test)]
del X_train
del X_test
print("Finish feature extraction")

event_files = [f"{base_path}/data/haenkaze/events.csv"]
event_files.append(f"{base_path}/data/haenkaze/event_range.csv")
score_file = f"result/haenkaze/{model_name}/{frequency}scores.csv"
#os.makedirs(score_file, exist_ok=True)
n_components = 10

# GMM training
isTrain = False
path_to_gmm = f"{base_path}/model/haenkaze/{model_name}"
os.makedirs(path_to_gmm, exist_ok=True)
gmm_model = f"{path_to_gmm}/gmm_{frequency}.pkl"
if os.path.exists(gmm_model) and not isTrain:
    print(f"Load from {gmm_model}")
    with open(gmm_model, "rb") as file:
        gmm = pickle.load(file)
else:
    print("GMM Training")
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42, n_init=10, max_iter=25)
    gmm.fit(feature_train)
    print("Finish GMM Training")
    with open(gmm_model, "wb") as file:
        pickle.dump(gmm, file)
    print(f"Model saved as {gmm_model}")  

log_likelihood = -gmm.score_samples(feature_train)
threshold = np.percentile(log_likelihood,threshold_line)
#threshold = max(log_likelihood)
del feature_train
anomaly_score = -gmm.score_samples(feature_test)
#import pdb; pdb.set_trace()
df = pd.DataFrame({"DATETIME": timestamp, "AnomalyScore": anomaly_score})
if(not(pd.api.types.is_datetime64_any_dtype(df['DATETIME']))):
    df['DATETIME'] = pd.to_datetime(df['DATETIME'],format='%d/%m/%y %H')
# 外れ値の除去
num_remove = 15
# AnomalyScore列の上位num_remove個のインデックスと値を取得
remove_indices = df.nlargest(num_remove, 'AnomalyScore').index
# データフレームから削除
df = df.drop(remove_indices) 
#df['Date'] = df['DATETIME'].dt.date

#import pdb; pdb.set_trace()

#img_path = out_dir + "/"+frequency+"_gmm_" + str(n_components) + "compsonents.png"
log_plot = False
if log_plot:img_path = f"{out_dir}/{frequency}_log.png"
else:img_path = f"{out_dir}/{frequency}.png"
plot_by_date(log_plot,anomaly_score,timestamp,train_range,threshold,img_path,event_files)
print(f"Threshold line: {threshold_line}%")
calc_scores(df, threshold, threshold_line, frequency, [event_files[0]], score_file, before_max=7)
print(f"image result: {img_path}")
print(f"Score result: {score_file}")