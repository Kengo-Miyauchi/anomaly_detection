import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from util_module import SCADA_utils
import re
import csv

# set device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# config
stampcol = "DateTime"
dataset_name = "haenkaze"
year = "2023"
data_path = "/mnt/iot-qnap5/miyauchi"
data_dir = f'{data_path}/data/{dataset_name}/{year}/1s_data'
result_file = f'{data_path}/data/paired_txt.csv'
paired_df_dir = f'{data_path}/data/paired_df'

# 保存ディレクトリがなければ作成
os.makedirs(paired_df_dir, exist_ok=True)

# load files
file_list = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir)])
print(f"Data directory: {data_dir}")

with open(result_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Text"])  # ヘッダー行

    for file_idx, file_path in enumerate(file_list):
        match = re.search(r'(\d{8})', file_path)
        if not match:
            continue
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        try:
            df = pd.read_csv(file_path, encoding='shift-jis', skipfooter=1, engine='python')
        except Exception as e:
            print(f"Error in file {file_path}: {e}")
            continue

        if df.shape[1] <= 1:
            print(f"Data {file_path}: only DateTime")
            continue

        print(f"{file_idx+1}/{len(file_list)} processing... {file_path}")

        df[stampcol] = pd.to_datetime(df[stampcol])
        SCADA_utils.fix_data(df)
        df.fillna(method="ffill", inplace=True)
        SCADA_utils.arrange_data(df, stampcol)

        start_time = df[stampcol].min().replace(minute=0, second=0)
        end_time = df[stampcol].max()

        i = 0
        while start_time + timedelta(hours=1) <= end_time:
            end_hour = start_time + timedelta(hours=1)
            df_hour = SCADA_utils.extract_specific_terms(df, start_time, end_hour, stampcol)

            if df_hour.empty:
                start_time = end_hour
                i += 1
                continue

            relative_changes = {}
            directions = {}

            for col in df_hour.columns:
                if col == stampcol:
                    continue
                values = df_hour[col].dropna()
                if len(values) < 2:
                    continue
                start_val = values.iloc[0]
                end_val = values.iloc[-1]
                mean_val = values.mean()
                if mean_val == 0:
                    continue
                delta = end_val - start_val
                rel_change = delta / mean_val
                relative_changes[col] = abs(rel_change)
                directions[col] = "増加" if rel_change > 0 else "減少"

            if not relative_changes:
                start_time = end_hour
                i += 1
                continue

            most_changed_attr = max(relative_changes, key=relative_changes.get)
            direction_text = directions[most_changed_attr]
            hour_str = f"{start_time.strftime('%H:%M')}~{end_hour.strftime('%H:%M')}"
            date_id = f"{start_time.strftime('%y-%m-%d')}-{i}"
            text = f"{hour_str}で{most_changed_attr}の値が大きく{direction_text}した"

            # テキスト出力
            writer.writerow([date_id, text])

            # df_hourの保存
            df_hour_path = os.path.join(paired_df_dir, f"{date_id}.csv")
            df_hour.to_csv(df_hour_path, index=False, encoding='utf-8')

            start_time = end_hour
            i += 1

print(f"result saved to {result_file}")
print(f"hourly data saved to {paired_df_dir}")
