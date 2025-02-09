import torch
import pandas as pd
import numpy as np
import os
import gc
from util_module.data_to_plot import plot_by_date

from wind_module import utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
frequency = '1S'
use_pca = False


# download data
#frequency = '10S_sampled'
dataset_name = "haenkaze"
stampcol = "DateTime"
if(frequency=='1S' or 'sampled'in frequency):
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_data.parquet'
else:
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_avg_data.parquet'

print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
#data = utils.arrange_data(data,stampcol)
print("Finished loading data.")

# preprocessing
#data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2023-04-01','2023-04-30')
test_range = ('2023-04-01','2023-09-30')
train = utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
test = utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
timestamp = test[stampcol]
del data

if not use_pca:
    X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
    del train
    X_test = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
    del test
    # memory management
    gc.collect()
    X_train = X_train.astype(np.float16)
    X_test = X_test.astype(np.float16)

event_files = ["data/haenkaze/events.csv"]
event_files.append("data/haenkaze/event_range.csv")
score_file = "result/haenkaze/gmm-only/"+frequency+"scores.csv"
n_components=10
covariance_type="full"
log_plot=False
out_dir="result/haenkaze/gmm-only"

from sklearn.mixture import GaussianMixture
import pickle
#print("GMM Training")
if use_pca:
    from sklearn.decomposition import PCA
    #主成分分析の実行
    pca = PCA()
    train = train.iloc[:, 0:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    pca.fit(train)
    feature_train = pca.transform(train)
    del train
    feature_train = feature_train[:,:,40]
    test = test.iloc[:, 0:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    feature_test = pca.transform(test)
    del test
    feature_test = feature_test[:,:40]
else:
    feature_train=X_train
    feature_test=X_test

gmm_model = f"model/haenkaze/gmm/{frequency}.pkl"
if os.path.exists(gmm_model):
    print(f"Load from {gmm_model}")
    with open(gmm_model, "rb") as file:
        gmm = pickle.load(file)
else:
    print("GMM Training")
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42, n_init=10, max_iter=25)
    gmm.fit(feature_train)
    print("Finish GMM Training")
    with open(gmm_model, "wb") as file:
        pickle.dump(gmm, file)
    print("Model saved as gmm_model.pkl")  
log_likelihood = -gmm.score_samples(feature_train)
threshold = np.percentile(log_likelihood,99.9)
del feature_train
print("Finish GMM Training")
img_path = out_dir + "/"+frequency+"gmm_" + str(n_components) + "components.png"
anomaly_score = -gmm.score_samples(feature_test)
del feature_test
plot_by_date(log_plot,anomaly_score,timestamp,train_range,threshold,img_path,event_files)
print(f"result: {img_path}")
print(f"Score result: {score_file}")