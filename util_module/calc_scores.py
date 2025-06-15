import pandas as pd
import csv
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.special import expit
import csv

def get_date_range(start,end):
    dates = []
    current_date = start
    while current_date <= end:
        date = current_date.date()
        dates.append(date)
        current_date += timedelta(days=1)
    return dates

def get_event_range(event_files, before):
    event_dates = set()
    for event_file in event_files:
        events=pd.read_csv(event_file)
        if (events.shape[1]==1):type="line"
        else: type="range"
        if(type=="line"):
            dates = events["event_date"].values
            for date in dates:
                start = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=before)
                end = datetime.strptime(date, "%Y-%m-%d")
                event_dates.update(get_date_range(start,end))
        elif(type=="range"):
            start_date=events['start_date'].values
            end_date=events['end_date'].values
            for i in range(len(start_date)):
                start = datetime.strptime(start_date[i], "%Y-%m-%d") - timedelta(days=before)
                end = datetime.strptime(end_date[i], "%Y-%m-%d")
                event_dates.update(get_date_range(start,end))
    return list(event_dates)

def get_anomaly_dates(event_files, before):
    anomaly_dates = set()
    for event_file in event_files:
        events=pd.read_csv(event_file)
        if (events.shape[1]==1):type="line"
        else: type="range"
        if(type=="line"):
            event_dates = events["event_date"].values
            for date in event_dates:
                anomaly_date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=before)
                anomaly_dates.add(anomaly_date.date())
        elif(type=="range"):
            start_date=events['start_date'].values
            end_date=events['end_date'].values
            for i in range(len(start_date)):
                start = datetime.strptime(start_date[i], "%Y-%m-%d") - timedelta(days=before)
                end = datetime.strptime(end_date[i], "%Y-%m-%d")
                anomaly_dates.update(get_date_range(start,end))
    return list(anomaly_dates)

def get_normal_dates(event_files, before, range_days, anomaly_dates):
    normal_dates = set()
    anomaly_dates = set(anomaly_dates)
    for event_file in event_files:
        events = pd.read_csv(event_file)
        if events.shape[1]==1: type = "line"
        else: type = "range"
        if type == "line":
            event_dates = events["event_date"].values
            for date in event_dates:
                event_date = datetime.strptime(date, "%Y-%m-%d")
                start = event_date - timedelta(days=before+range_days)
                end = event_date - timedelta(days=before+1)
                for day in get_date_range(start, end):
                    if day not in anomaly_dates:
                        normal_dates.add(day)
        elif type == "range":
            start_dates = events['start_date'].values
            for i in range(len(start_dates)):
                start = datetime.strptime(start_dates[i], "%Y-%m-%d") - timedelta(days=before + range_days)
                end = datetime.strptime(start_dates[i], "%Y-%m-%d") - timedelta(days=before+1)
                for day in get_date_range(start, end):
                    if day not in anomaly_dates:
                        normal_dates.add(day)
    return sorted(list(normal_dates))



def calc_scores(df, threshold, threshold_line, frequency, event_files, score_file, before_max):
    precision_scores = ["precision"]
    recall_scores = ["recall"]
    f1_scores = ["f1"]
    auc_scores = ["auc"]

    df['Date'] = df['DATETIME'].dt.date
    anomaly_dates_full = get_event_range(event_files, before_max)
    normal_dates = get_normal_dates(event_files, before_max, before_max * 3, anomaly_dates_full)
    df_normal = df[df["Date"].isin(normal_dates)].copy()
    df_normal["Predicted"] = (df_normal["AnomalyScore"] > threshold).astype(int)
    df_normal["Label"] = 0

    for before in range(before_max + 1):
        anomaly_dates = get_anomaly_dates(event_files, before)
        df_anomaly = df[df["Date"].isin(anomaly_dates)].copy()
        df_anomaly["Predicted"] = (df_anomaly["AnomalyScore"] > threshold).astype(int)
        df_anomaly["Label"] = 1
        df_all = pd.concat([df_normal, df_anomaly])
        y_true = df_all["Label"]
        y_pred = df_all["Predicted"]
        #import pdb; pdb.set_trace()

        precision = '{:.2f}'.format(precision_score(y_true, y_pred))
        recall = '{:.2f}'.format(recall_score(y_true, y_pred))
        f1 = '{:.2f}'.format(f1_score(y_true, y_pred))
        
        raw_scores = df_all["AnomalyScore"].values
        mean = raw_scores.mean()
        std = raw_scores.std() if raw_scores.std() > 0 else 1
        scaled_scores = (raw_scores - mean) / std
        roc_scores = expit(scaled_scores)
        try:
            auc = '{:.2f}'.format(roc_auc_score(y_true, roc_scores))
        except ValueError:
            auc = "NA"  # 片方のクラスしか存在しない場合
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

    with open(score_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([frequency, f"threshold:{threshold_line}"])
        writer.writerow(precision_scores)
        writer.writerow(recall_scores)
        writer.writerow(f1_scores)
        writer.writerow(auc_scores)
    print(f"Score result: {score_file}.csv")

def calc_scores_with_windowing(df, threshold, threshold_line, frequency, event_files, score_file, before_max, window='1min'):  
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df['Date'] = df['DATETIME'].dt.date

    anomaly_dates_full = get_event_range(event_files, before_max)
    normal_dates = get_normal_dates(event_files, before_max, before_max * 3, anomaly_dates_full)

    precision_scores = ["precision"]
    recall_scores = ["recall"]
    f1_scores = ["f1"]
    auc_scores = ["auc"]

    for before in range(before_max + 1):
        anomaly_dates = get_anomaly_dates(event_files, before)

        # 正常・異常のデータを抽出
        df_anomaly = df[df["Date"].isin(anomaly_dates)].copy()
        df_normal = df[df["Date"].isin(normal_dates)].copy()

        df_anomaly["Label"] = 1
        df_normal["Label"] = 0

        # 時系列としてインデックスに変換
        df_anomaly['DATETIME'] = pd.to_datetime(df_anomaly['DATETIME'])
        df_normal['DATETIME'] = pd.to_datetime(df_normal['DATETIME'])
        df_anomaly = df_anomaly.set_index('DATETIME')
        df_normal = df_normal.set_index('DATETIME')

        # resampleは数値列のみに明示的に適用する（AnomalyScoreのみ）
        df_anomaly_r = df_anomaly[['AnomalyScore']].resample(window).mean().dropna()
        df_anomaly_r['Label'] = 1

        df_normal_r = df_normal[['AnomalyScore']].resample(window).mean().dropna()
        df_normal_r['Label'] = 0

        # 結合
        df_all = pd.concat([df_anomaly_r, df_normal_r])
        df_all["Predicted"] = (df_all["AnomalyScore"] > threshold).astype(int)

        # 評価指標計算
        y_true = df_all["Label"]
        y_pred = df_all["Predicted"]

        precision = '{:.2f}'.format(precision_score(y_true, y_pred, zero_division=0))
        recall = '{:.2f}'.format(recall_score(y_true, y_pred, zero_division=0))
        f1 = '{:.2f}'.format(f1_score(y_true, y_pred, zero_division=0))

        raw_scores = df_all["AnomalyScore"].values
        mean = raw_scores.mean()
        std = raw_scores.std() if raw_scores.std() > 0 else 1
        scaled_scores = (raw_scores - mean) / std
        roc_scores = expit(scaled_scores)
        try:
            auc = '{:.2f}'.format(roc_auc_score(y_true, roc_scores))
        except ValueError:
            auc = "NA"

        # 記録
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

    # 書き出し
    with open(f"{score_file}.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([frequency, f"threshold:{threshold_line}", f"window:{window}"])
        writer.writerow(precision_scores)
        writer.writerow(recall_scores)
        writer.writerow(f1_scores)
        writer.writerow(auc_scores)
    print(f"Score result: {score_file}_{window}.csv")

def calc_scores_windowing_once(df, threshold, threshold_line, frequency, event_files, score_file, before_max, window='1min'):

    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df['Date'] = df['DATETIME'].dt.date

    anomaly_dates_full = get_event_range(event_files, before_max)
    normal_dates = get_normal_dates(event_files, before_max, before_max * 3, anomaly_dates_full)

    precision_scores = ["precision"]
    recall_scores = ["recall"]
    f1_scores = ["f1"]
    auc_scores = ["auc"]

    for before in range(before_max + 1):
        anomaly_dates = get_anomaly_dates(event_files, before)

        # 正常・異常データを抽出
        df_anomaly = df[df["Date"].isin(anomaly_dates)].copy()
        df_anomaly["Label"] = 1
        df_normal = df[df["Date"].isin(normal_dates)].copy()
        df_normal["Label"] = 0

        # 時系列インデックス化
        df_anomaly = df_anomaly.set_index('DATETIME')[['AnomalyScore', 'Label']]
        df_normal = df_normal.set_index('DATETIME')[['AnomalyScore', 'Label']]

        # max() を使って「1回でも超えたら異常」とする
        df_anomaly_r = df_anomaly['AnomalyScore'].resample(window).max().dropna().to_frame()
        df_anomaly_r['Label'] = 1

        df_normal_r = df_normal['AnomalyScore'].resample(window).max().dropna().to_frame()
        df_normal_r['Label'] = 0

        # 結合
        df_all = pd.concat([df_anomaly_r, df_normal_r])

        # 一度でも超えたら異常と判定
        df_all["Predicted"] = (df_all["AnomalyScore"] > threshold).astype(int)

        # 評価指標計算
        y_true = df_all["Label"]
        y_pred = df_all["Predicted"]

        precision = '{:.2f}'.format(precision_score(y_true, y_pred, zero_division=0))
        recall = '{:.2f}'.format(recall_score(y_true, y_pred, zero_division=0))
        f1 = '{:.2f}'.format(f1_score(y_true, y_pred, zero_division=0))

        # AUCスコア算出（スケーリング+sigmoid）
        raw_scores = df_all["AnomalyScore"].values
        mean = raw_scores.mean()
        std = raw_scores.std() if raw_scores.std() > 0 else 1
        scaled_scores = (raw_scores - mean) / std
        roc_scores = expit(scaled_scores)
        try:
            auc = '{:.2f}'.format(roc_auc_score(y_true, roc_scores))
        except ValueError:
            auc = "NA"

        # 記録
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

    # 結果保存
    with open(f"{score_file}_{window}_once.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([frequency, f"threshold:{threshold_line}", f"window:{window}", "mode: max_in_window"])
        writer.writerow(precision_scores)
        writer.writerow(recall_scores)
        writer.writerow(f1_scores)
        writer.writerow(auc_scores)
    print(f"Score result: {score_file}_{window}_once.csv")