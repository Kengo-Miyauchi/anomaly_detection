import pandas as pd
import os
# download dataset
dataset_name = "haenkaze"
year="2023"
data_dir = 'data/' + dataset_name + '/'+year+'/1s_data'
parquet_file = f'data/{dataset_name}/'+year+'_data.parquet'
num_frame = 3
framed_file = f'data/{dataset_name}/'+year+'_'+str(num_frame)+'frame_data.parquet'

if os.path.exists(parquet_file):
    print("Loading data from parquet_file...")
    base = pd.read_parquet(parquet_file)
    print("Finish loading data...")
    df_cp = base.drop(columns=['DateTime',' 日付ﾌｫｰﾏｯﾄ 時分秒'])
    for i in range(num_frame):
        df = df_cp[i+1:]
        df.columns = [f"{col}_{i+1}" for col in df.columns]
        base = pd.concat([base, df], axis=1)
        del df
    base = base[:-num_frame+1]
    base.to_parquet(framed_file, engine='pyarrow')
    print(f"Data saved in Parquet format: {framed_file}")
else:print(f"{parquet_file} does not exsist.")