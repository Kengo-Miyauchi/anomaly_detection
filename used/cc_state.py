import pandas as pd
import os
import matplotlib.pyplot as plt

# --- 設定 ---
dataset_name = "haenkaze"
base_dir = "/mnt/work-qnap/miyauchi/data"
year_list = ["2020", "2021", "2022", "2023", "2024"]

cc_col = " CC状態(Real)"
stampcol = "DateTime"

output_dir = f"result/{dataset_name}/cc_state_plot"
os.makedirs(output_dir, exist_ok=True)

for year in year_list:
    print(f"\n=== {year}年の処理を開始 ===")
    year_dir = os.path.join(base_dir, dataset_name, year, "1s_data")
    if not os.path.exists(year_dir):
        print(f"→ フォルダが存在しません: {year_dir}")
        continue

    # 月ごとのデータ格納用
    monthly_data = {}

    file_list = sorted([os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith('.csv')])
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, encoding='shift-jis', skipfooter=1, engine='python')
            if stampcol in df.columns and cc_col in df.columns:
                df[stampcol] = pd.to_datetime(df[stampcol], errors='coerce')
                df = df[[stampcol, cc_col]].dropna()
                df[cc_col] = df[cc_col].astype(int)

                # 月ごとに分類
                df['month_key'] = df[stampcol].dt.strftime('%Y-%m')
                for month, group in df.groupby('month_key'):
                    if month not in monthly_data:
                        monthly_data[month] = []
                    monthly_data[month].append(group[[stampcol, cc_col]])
            else:
                print(f"→ カラム不足: {file_path}")
        except Exception as e:
            print(f"→ 読み込みエラー: {file_path}: {e}")

    # --- 月ごとのプロット ---
    for month_key, df_list in monthly_data.items():
        df_month = pd.concat(df_list, ignore_index=True).dropna()
        df_month.sort_values(by=stampcol, inplace=True)

        if df_month.empty:
            continue

        # プロット
        plt.figure(figsize=(15, 4))
        plt.plot(df_month[stampcol], df_month[cc_col], color='blue', linewidth=0.3)
        plt.title(f"CC状態の1秒時系列変動 - {month_key}")
        plt.xlabel("Time")
        plt.ylabel("CC状態")
        plt.grid(True)
        plt.tight_layout()

        # 保存先ディレクトリとパス
        year_str, month_str = month_key.split('-')
        month_dir = os.path.join(output_dir, year_str, month_str)
        os.makedirs(month_dir, exist_ok=True)

        save_path = os.path.join(month_dir, f"{month_key}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"✅ 保存完了: {save_path}")
