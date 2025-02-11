import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta

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