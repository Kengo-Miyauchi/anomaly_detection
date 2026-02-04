import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime
from util_module import SCADA_utils
import re
import csv

# set device
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# download data
stampcol = "DateTime"
frequency = '1S'
sequence_length = 10
threshold_line = 100 - 1
dataset_name = "haenkaze"
data_path = "/mnt/work-qnap/miyauchi"

dataset_name = "haenkaze"
year="2023"
data_dir = f'{data_path}/data/{dataset_name}/{year}/1s_data'
file_list = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir)])
print(f"Data directory: {data_dir}")

# TabNet モデルの読み込み
path_to_pretrained = "model/haenkaze/tabnet-pretrain-out2023-40dim"
#path_to_pretrained = f"model/haenkaze/tabnet-self-attn-40dim-10seq"
unsupervised_model = torch.load(path_to_pretrained+"/pretrained.pth",weights_only=False)

top_k = 5
result_file = f'top{top_k}_features.csv'
# 正常時のデータとの差分がある特徴量について、そのAttentionを見たい
with open(result_file, 'w') as f:
    writer = csv.writer(f)
    for file_idx, file_path in enumerate(file_list):
        match = re.search(r'(\d{8})', file_path)
        if match:
            date_str = match.group(1)
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        # YY/MM/DD形式に変換して出力
        date = date_obj.strftime("%y/%m/%d")
        # SCADAデータを1日ずつ抽出,前処理
        try:
            df = pd.read_csv(file_path, encoding='shift-jis', skipfooter=1, engine='python')
        except pd.errors.ParserError as e:
            print(f"ParserError in file {file_path}: {e}")
        except UnicodeDecodeError as e:
            print(f"Error in file: {file_path}")
            print(e)
        if(df.shape[1]==1):
            print(f"Data {file_path}: only DateTime")
        else:
            print(f"{file_idx+1}/{len(file_list)} processing...")
            df[stampcol] = pd.to_datetime(df[stampcol])
            SCADA_utils.fix_data(df)
            df.fillna(method="ffill")
            SCADA_utils.arrange_data(df,stampcol)
            data = df.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒'])
            feature_names = data.columns
            
            # 特徴量変換してimportanceを算出
            M_explain, masks = unsupervised_model.explain(data.values)
            mean_importance = np.mean(M_explain, axis=0)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_importance
            })
            importance_df = importance_df.sort_values(by='importance', ascending=False)
            top_features = importance_df.head(top_k)['feature'].values.tolist()
            top_features.insert(0, date)
            # 最重要の特徴量カラムをcsvに書き込む
            writer.writerow(top_features)
print(f"Top{top_k} features written to {result_file}")