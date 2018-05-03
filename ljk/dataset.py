#encoding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter 
# import xgboost as xgb
import datetime

def lag_format(t, i):
    return '{}_lag_{}'.format(t, i)

def create_lagging(df, df_original, i, targets):
    df1 = df_original.copy()
    df1['utc_time'] = df1['utc_time'] + pd.DateOffset(hours=i)
    df1 = df1.rename(columns={t: lag_format(t, i) for t in targets})
    df2 = pd.merge(df, df1[['stationId', 'utc_time', 'city'] + [lag_format(t, i) for t in targets]],
                    on=['stationId', 'utc_time', 'city'],
                    how='right')
    return df2

def read_raw(raw_path):
    raw_data = pd.read_csv(raw_path)
    # raw_data.drop(['weather'], axis=1, inplace=True)
    raw_data.weather = raw_data.weather.astype('category')
    raw_data.utc_time = pd.to_datetime(raw_data.utc_time)
    return raw_data

def create_dataset1(raw_path, MaxLagging = 3):
    raw_data = read_raw(raw_path)
    targets = ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed', 'CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'weather']
    df1 = create_lagging(raw_data, raw_data, 1, targets)
    for i in range(2, MaxLagging + 1):
        print('lagging: %d'%i)
        df1 = create_lagging(df1, raw_data, i, targets)
    df1.hour = df1.utc_time.dt.hour
    return df1.sort_values(by=['utc_time', 'stationId'])