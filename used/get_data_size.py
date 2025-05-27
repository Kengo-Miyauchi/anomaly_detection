import torch
import pandas as pd
import os
from util_module import SCADA_utils

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"b
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download data
stampcol = "DateTime"
frequency_list =['10M','1M','10S','10S_sampled','1S']
dataset_name = "haenkaze"
for frequency in frequency_list:
    if(frequency=='1S' or 'sampled' in frequency):
        parquet_file = f'data/{dataset_name}/2023_'+frequency+'_data.parquet'
    else:
        parquet_file = f'data/{dataset_name}/2023_'+frequency+'_avg_data.parquet'
    data = pd.read_parquet(parquet_file)
    #data = utils.fix_data(data)
    #data = utils.arrange_data(data,stampcol)
    print(f"{frequency} Data size: {len(data)}")
    del data