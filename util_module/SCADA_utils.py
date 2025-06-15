import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
import statistics
from typing import List, Union

def arrange_data(df,stampcol):
    df = df.fillna(method="ffill")
    df[stampcol] = pd.to_datetime(df[stampcol])
    sorted_df = df.sort_values(by=stampcol)
    return sorted_df

def fix_data(df):
    df_cleaned = df.applymap(
        lambda x: 
            np.nan if isinstance(x, str) and (x.strip() == '' or x.strip() == '-')
            else float(x.replace(' ', '')) if isinstance(x, str) and x.strip().replace('.', '', 1).lstrip('-').isdigit()
            else x
    )
    return df_cleaned

def extract_specific_id(df,id):
    data = df[df['Turbine_ID'] == id]
    return data

def extract_specific_terms(df,start,end,stampcol):
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    filtered_df = df[(df[stampcol] >= start_time) & (df[stampcol] <= end_time)]
    return filtered_df

def get_months(df,stampcol):
    data_by_month = df.groupby(df[stampcol].dt.to_period("M")).size()
    return data_by_month

def new_time(time,n,type):
    if(type=="months"):
        new_time = time + relativedelta(months=n)
    if(type=="minutes"):
        span = 10
        new_time = time + timedelta(minutes=n * span)
    return new_time

def median_timestamps(
    timestamps: List[Union[str, datetime]],
    sequence_length: int,
    time_fmt: str = None
) -> List[datetime]:
    """
    1次元配列のタイムスタンプを sequence_length ごとにブロック化し、
    各ブロックの中央値(timestamp)を返す。

    Parameters
    ----------
    timestamps : list of str or datetime
        元のタイムスタンプ列。文字列の場合は time_fmt で parse します。
    sequence_length : int
        ブロックの長さ。
    time_fmt : str, optional
        文字列タイムスタンプを parse するフォーマット（例: "%Y-%m-%d %H:%M:%S"）。
        None の場合は datetime.fromisoformat を試みます。

    Returns
    -------
    List[datetime]
        各入力に対応するブロック中央値のタイムスタンプ（長さは元と同じ）。
    """
    # --- 1) 入力を datetime のリストに変換 ---
    dt_list: List[datetime] = []
    for t in timestamps:
        if isinstance(t, datetime):
            dt_list.append(t)
        else:
            # 文字列の場合
            if time_fmt:
                dt_list.append(datetime.strptime(t, time_fmt))
            else:
                dt_list.append(datetime.fromisoformat(t))

    n = len(dt_list)
    medians: List[datetime] = [None] * n

    # --- 2) ブロック単位で中央値を計算 ---
    for start in range(0, n, sequence_length):
        block = dt_list[start:start + sequence_length]
        # ブロック内を UNIX 秒に変換し中央値を取る
        secs = [d.timestamp() for d in block]
        med_sec = statistics.median(secs)
        med_dt = datetime.fromtimestamp(med_sec)

        # ブロック内の各位置に同じ中央値を割り当て
        for idx in range(start, start + len(block)):
            medians[idx] = med_dt

    return medians