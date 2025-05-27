from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime

def visualize_events(event_file):
    event_dates=pd.read_csv(event_file)
    if (event_dates.shape[1]==1):type="line"
    else: type="range"
    if(type=="line"):
        events = event_dates["event_date"]
        for i in range(len(events)):
            event = pd.Timestamp(events[i])
            plt.axvline(event, color='orange', alpha=1.0)
    elif(type=="range"):
        start_date=event_dates['start_date']
        end_date=event_dates['end_date']
        # イベント区間の可視化
        for i in range(len(start_date)):
            start = datetime.strptime(start_date[i], "%Y-%m-%d")
            end = datetime.strptime(end_date[i], "%Y-%m-%d")
            plt.axvspan(start, end, color='orange', alpha=0.2)

def color_train_range(train_range):
    plt.axvspan(train_range[0], train_range[1], color='green', alpha=0.2)

def plot_by_date(log_plot,anomaly_score,timestamp,train_range,threshold,img_path,event_files=None):
    df = pd.DataFrame({"DATETIME": timestamp, "AnomalyScore": anomaly_score})
    if(not(pd.api.types.is_datetime64_any_dtype(df['DATETIME']))):
        df['DATETIME'] = pd.to_datetime(df['DATETIME'],format='%d/%m/%y %H')
    # 散布図をプロット
    plt.figure(figsize=(32, 10))
    
    if log_plot:
        plt.yscale('log')
    """df['Date'] = df['DATETIME'].dt.date 
    for date, group in df.groupby('Date'):
        plt.scatter([date]*len(group), group['AnomalyScore'], alpha=0.8, marker='o', color='blue') """
        
    # 外れ値の除去
    num_remove = 15
    # AnomalyScore列の上位num_remove個のインデックスと値を取得
    remove_indices = df.nlargest(num_remove, 'AnomalyScore').index
    """ removed_values = df.loc[remove_indices, 'AnomalyScore']
    # 削除対象のデータを記録
    removed_data = df.loc[remove_indices] """
    # データフレームから削除
    df = df.drop(remove_indices)
    
    plt.scatter(df['DATETIME'], df['AnomalyScore'],alpha=0.8, marker='o', color='blue')
    plt.xlabel("Date",fontsize=40)
    plt.ylabel("Anomaly Score",fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=28)
    plt.title("Anomaly Scores by Date")
    #plt.xticks(rotation=45)
    plt.grid(True)
    plt.axhline(y=threshold, color='r', linestyle='solid')
    
    # 学習期間/イベント情報の可視化
    color_train_range(train_range)
    if(event_files!=None):
        for file in event_files:
            visualize_events(file)
    
    # 凡例の表示を防止するための設定（重複する日付を削除）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(img_path)
    plt.clf()
    plt.close()