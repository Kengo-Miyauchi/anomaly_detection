import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from util_module import SCADA_utils
from sklearn.preprocessing import StandardScaler

stampcol = "DateTime"
matplotlib.rcParams['font.family'] = 'IPAexGothic'
scaler = StandardScaler()

def heatmap(df, img_path):
    # データのスケーリング
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 0~1の範囲にスケーリング
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    # 標準化
    #df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].std())
    # タイムスタンプをインデックスに設定
    df.set_index(stampcol, inplace=True)
    # データの転置 (縦軸: 属性, 横軸: 時間)
    df_transposed = df.T
    # ヒートマップの描画
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        df_transposed,
        cmap="viridis",          # カラーマップ（変更可能）
        annot=False,             # 値を表示したい場合はTrueに
        cbar=True,               # カラーバーを表示
        xticklabels=True,        # 時刻のラベルを表示
        yticklabels=True         # 属性名のラベルを表示
    )
    ax.set_xticks([])
    ax.set_yticks([])
    step_size = 5
    yticks = np.arange(0, len(df_transposed.index), step=step_size)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    # ラベルの設定
    plt.title("Time-Attribute Heatmap")
    plt.xlabel("Time")
    plt.ylabel("Attributes")
    #plt.xticks(rotation=45, ha="right")  # 時刻ラベルを回転
    plt.tight_layout()
    plt.show()
    plt.savefig(img_path)

start_time = "01:00:00"
end_time = "03:00:00"

stampcol = "DateTime"
date = "2023-04-11"
normal_data = "data/haenkaze/2023/1s_data/Analog_20230411_000000.csv"
df_normal = pd.read_csv(normal_data, encoding='shift-jis', skipfooter=1, engine='python')
df_normal[stampcol] = pd.to_datetime(df_normal[stampcol])
df_normal = SCADA_utils.fix_data(df_normal)
#print(df_normal.shape[1])
start = date + " " + start_time
end = date + " " + end_time
df_normal = SCADA_utils.extract_specific_terms(df_normal,start,end,stampcol)
img_path = "heatmap/Normal_heatmap"+date+"-scaled.png"
heatmap(df_normal,img_path)
print(f"Normal Heatmap: {img_path}")

date = "2023-05-27"
anomaly_data = "data/haenkaze/2023/1s_data/Analog_20230527_000000.csv"
df_anomaly = pd.read_csv(anomaly_data, encoding='shift-jis', skipfooter=1, engine='python')
df_anomaly[stampcol] = pd.to_datetime(df_anomaly[stampcol])
df_anomaly = SCADA_utils.fix_data(df_anomaly)
start = date + " " + start_time
end = date + " " + end_time
df_anomaly = SCADA_utils.extract_specific_terms(df_anomaly,start,end,stampcol)
img_path = "heatmap/Anomaly_heatmap"+date+"-scaled.png"
heatmap(df_anomaly,img_path)
print(f"Anomaly Heatmap: {img_path}")
