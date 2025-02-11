import torch
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
# download dataset
dataset_name = "haenkaze"
year="2020"
data_dir = 'data/' + dataset_name + '/'+year+'/1s_data'
parquet_file = f'data/{dataset_name}/'+year+'_data.parquet'

# check cache file
if os.path.exists(parquet_file):
    print("Loading data from parquet_file...")
    data = pd.read_parquet(parquet_file)
else:
    print("Loading data from CSV files...")
    dfs = []
    file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
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
            dfs.append(df)
            print(f"Download {file_idx+1}/{len(file_list)} completed")
    data = pd.concat(dfs, ignore_index=True)
    data.to_parquet(parquet_file, engine='pyarrow')
    print("Data saved in Parquet format.")
print("Finished loading data.")