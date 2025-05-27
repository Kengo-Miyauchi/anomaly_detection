import torch
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from util_module import SCADA_utils

# download dataset
dataset_name = "haenkaze"
year="2023"
frequency = input("frequency: ")
data_dir = 'data/' + dataset_name + '/'+year+'/1s_data'
parquet_file = f'data/{dataset_name}/'+year+'_'+frequency+'_fixed.parquet'

stampcol = "DateTime"
dfs = []
file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]


print(f"file path: {parquet_file}")
for file_idx, file_path in enumerate(file_list):
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
        df[stampcol] = pd.to_datetime(df[stampcol])
        SCADA_utils.fix_data(df)
        df.fillna(method="ffill")
        dfs.append(df)
        print(f"Download {file_idx+1}/{len(file_list)} completed")
data = pd.concat(dfs, ignore_index=True)
data.to_parquet(parquet_file, engine='pyarrow')