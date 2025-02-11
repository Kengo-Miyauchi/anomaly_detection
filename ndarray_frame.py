import torch
import pandas as pd
import numpy as np
import os
import gc
from util_module.data_to_plot import plot_by_date, calc_scores
from util_module.end_info import show_info
from util_module.tabnet.tabnet_feature import create_dataloader, data_to_framedFeatures
from util_module.tabnet.build_exec_model import ExecModel
from util_module.tabnet.set_config import set_config_file
from util_module import SCADA_utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download dataset
dataset_name = "haenkaze"
stampcol = "DateTime"
frequency = '1S'
isTrain = True
num_frame = 5
threshold_line = 100 - 0.001
if(frequency=='1S' or 'sampled' in frequency):
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_data.parquet'
else:
    parquet_file = f'data/{dataset_name}/2023_'+frequency+'_avg_data.parquet'
print(f"Loading data from {parquet_file}...")
data = pd.read_parquet(parquet_file)
data = SCADA_utils.arrange_data(data,stampcol)
print("Finish loading data.")

# preprocessing
#data[stampcol] = pd.to_datetime(data[stampcol])
train_range = ('2023-04-01','2023-04-30')
test_range = ('2023-04-01','2023-09-30')
train = SCADA_utils.extract_specific_terms(data,train_range[0],train_range[1],stampcol)
test = SCADA_utils.extract_specific_terms(data,test_range[0],test_range[1],stampcol)
timestamp = test[stampcol]

X_train = train.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del train
gc.collect()
#import pdb; pdb.set_trace()
X_test = test.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒']).values
del test
gc.collect()


# set execute model
model_name = f"tabnet-gmm-{num_frame}frame"
config = set_config_file()
exec_model = ExecModel(device,config,dataset_name,model_name,X_train)
out_dir = exec_model.out_dir


# convert data to tabnet encoder features
print("Start Feature Extraction")
feature_train = data_to_framedFeatures(exec_model,X_train,num_frame,True)
del X_train
print("Finish Feature Extraction")

from sklearn.mixture import GaussianMixture
import numpy as np
import pickle
n_components = 10
path_to_gmm = f"model/haenkaze/{model_name}"
os.makedirs(path_to_gmm, exist_ok=True)
gmm_model = f"{path_to_gmm}/gmm_{frequency}.pkl"
if os.path.exists(gmm_model) and (not isTrain):
    print(f"Load from {gmm_model}")
    with open(gmm_model, "rb") as file:
        gmm = pickle.load(file)
else:
    print("GMM Training")
    gmm = GaussianMixture(n_components=n_components, covariance_type=exec_model.covariance_type, random_state=42, n_init=10, max_iter=25)
    gmm.fit(feature_train)
    print("Finish GMM Training")
    with open(gmm_model, "wb") as file:
        pickle.dump(gmm, file)
    print(f"Model saved as {gmm_model}")  
log_likelihood = -gmm.score_samples(feature_train)
threshold = np.percentile(log_likelihood,threshold_line)
del feature_train, log_likelihood

print("Start Test Calculation")
#feature_test = data_to_framedFeatures(exec_model,X_test,num_frame,False)
dataloader = create_dataloader(X_test,exec_model.batch_size_tr,False)
# data to features as encoder output data
anomaly_score = []
for batch_index, batch in enumerate(dataloader):
    batch_output = []
    batch = batch.to(exec_model.device)
    try:
        step_outputs = exec_model.unsupervised_model.network.encoder(batch)[0]
    except Exception as e:
        print("\n"+f"Error occurred in batch {batch_index}: {e}")
        print(f"Batch shape: {batch.shape}")
        raise e
    encoder_out = sum(step for step in step_outputs)
    encoder_out = encoder_out.detach().cpu()
    del step_outputs, batch
    torch.cuda.empty_cache()
    i = 0
    #import pdb; pdb.set_trace()
    while((len(encoder_out)-i)>=num_frame):
        framed = []
        for j in range(num_frame):
            framed.extend(encoder_out[j])
        i+=1
        batch_output.append(framed)
    for j in range(num_frame-1):
        batch_output.append(framed)
    del framed
        #import pdb; pdb.set_trace()
    anomaly_score.extend(-gmm.score_samples(batch_output))
    del encoder_out, batch_output
    if ((batch_index+1)%100==0):print(f"{batch_index+1}/{len(dataloader)} batch:")
del X_test
print("Finish Test Calculation")

df = pd.DataFrame({"DATETIME": timestamp, "AnomalyScore": anomaly_score})
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
score_file = f"result/haenkaze/{model_name}/{frequency}scores.csv"
log_plot = False
if log_plot:img_path = f"{out_dir}/{frequency}_log.png"
else:img_path = f"{out_dir}/{frequency}.png"
plot_by_date(log_plot,anomaly_score,timestamp,train_range,threshold,img_path,event_files)
print(f"Threshold line: {threshold_line}%")
calc_scores(df, threshold, threshold_line, frequency, [event_files[0]], score_file, before_max=7)
print(f"result: {img_path}")
show_info(out_dir,exec_model)
print(f"Score result: {score_file}")