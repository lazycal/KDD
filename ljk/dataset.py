#encoding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter 
# import xgboost as xgb
import datetime

def lag_format(t, i):
    return '{}_lag_{}'.format(t, i) if i > 0 else t

def create_lagging(df, df_original, i, targets):
    df1 = df_original.copy()
    df1['utc_time'] = df1['utc_time'] + pd.DateOffset(hours=i)
    df1 = df1.rename(columns={t: lag_format(t, i) for t in targets})
    df2 = pd.merge(df, df1[['stationId', 'utc_time', 'city'] + [lag_format(t, i) for t in targets]],
                    on=['stationId', 'utc_time', 'city'],
                    how='right')
    return df2

def map_weather(aq_info):
    map_list = pd.read_csv('../data/rearranged/weather.csv')
    print(map_list)
    map_list = zip(map_list['1'], map(str, map_list['0']))
    map_list = dict(map_list)
    print(map_list)
    aq_info['weather_clu'] = aq_info.weather.map(lambda x: map_list[x] if x == x else np.nan)
    aq_info['weather_clu'] = aq_info['weather_clu'].astype('category')

def read_raw(raw_path):
    raw_data = pd.read_csv(raw_path)
    # raw_data.drop(['weather'], axis=1, inplace=True)
    raw_data.weather = raw_data.weather.astype('category')
    raw_data.utc_time = pd.to_datetime(raw_data.utc_time)
    for gas in ['PM2.5', 'PM10', 'O3']:
        raw_data[gas + '_raw'] = raw_data[gas]
    map_weather(raw_data)
    print(raw_data['weather_clu'])
    return raw_data

def create_dataset1(raw_path, MaxLagging = 3):
    raw_data = read_raw(raw_path)
    targets = ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed', 'CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'weather', 'weather_clu']
    df1 = create_lagging(raw_data, raw_data, 1, targets)
    for i in range(2, MaxLagging + 1):
        print('lagging: %d'%i)
        df1 = create_lagging(df1, raw_data, i, targets)
    df1.hour = df1.utc_time.dt.hour
    return df1.sort_values(by=['utc_time', 'stationId'])

def create_dataset5(raw_path, MaxLagging = 3):
    raw_data = read_raw(raw_path)
    for gas in ['PM2.5', 'PM10', 'O3']:
        raw_data[gas] = np.log1p(np.maximum(raw_data[gas], 0))
    targets = ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed', 'CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'weather', 'weather_clu']
    df1 = create_lagging(raw_data, raw_data, 1, targets)
    for i in range(2, MaxLagging + 1):
        print('lagging: %d'%i)
        df1 = create_lagging(df1, raw_data, i, targets)
    df1.hour = df1.utc_time.dt.hour
    return df1.sort_values(by=['utc_time', 'stationId'])