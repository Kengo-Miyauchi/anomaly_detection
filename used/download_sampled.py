import torch
import pandas as pd
import os
from util_module import SCADA_utils

freqency = '10S_sampled'
# frequency: 10S (10秒間隔)

# データセット情報
dataset_name = "haenkaze"
data_dir = f'data/{dataset_name}/2023/1s_data'
output_dir = f'data/{dataset_name}/2023/'+freqency+'sampled_data'
os.makedirs(output_dir, exist_ok=True)
final_output_file = f'data/{dataset_name}/2023_{freqency}_data.parquet'

# CSVファイルリストの取得
file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]


# 各ファイルでのサンプリング処理
def process_file(file_path):
    try:
        df = pd.read_csv(file_path, encoding='shift-jis', skipfooter=1, engine='python')
        timestamp_col = "DateTime"
        if timestamp_col not in df.columns:
            print(f"Warning: '{timestamp_col}' column not found in {file_path}. Skipping...")
            return None
        else:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col)  # タイムスタンプをインデックスに設定

        # 10秒ごとのサンプリング
        df_sampled = df.asfreq(freqency)  # 平均ではなく間隔ごとのデータを取得
        df_sampled = SCADA_utils.fix_data(df_sampled)
        df_sampled.fillna(method="ffill", inplace=True)  # 補間処理

        output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_'+freqency+'_sampled.parquet'))
        df_sampled.reset_index(inplace=True)  # インデックスをリセット
        df_sampled.to_parquet(output_file, engine='pyarrow')
        print(f"Processed and saved: {output_file}")
        return df_sampled
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


# 全ファイルを処理
all_sampled_dfs = []
for file_idx, file_path in enumerate(file_list):
    print(f"Processing file {file_idx + 1}/{len(file_list)}: {file_path}")
    sampled_df = process_file(file_path)
    if sampled_df is not None:
        all_sampled_dfs.append(sampled_df)

# 全ファイルの結果を結合して保存
if all_sampled_dfs:
    final_data = pd.concat(all_sampled_dfs, ignore_index=True)
    final_data.to_parquet(final_output_file, engine='pyarrow')
    print(f"All files processed and saved to {final_output_file}.")
else:
    print("No valid data to save.")
