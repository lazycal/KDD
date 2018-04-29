#encoding=utf-8
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as st
import numpy as np
import pyflux as pf
import matplotlib.pyplot as plt
import requests
# import xgboost as xgb
import datetime

def UpdateAqData():
    time_now = datetime.datetime.utcnow()
    time_text = time_now.strftime('20%y-%m-%d')
    start_time = time_text + '-0'
    end_time = time_text + '-23'
    time_period = start_time + '/' + end_time

    url = 'https://biendata.com/competition/airquality/bj/' + time_period + '/2k0d1d8'
    respones = requests.get(url)
    text = respones.text
    with open ("./data/temp/bj-data.csv", 'w') as f:
        f.write(text)
    new_info = pd.read_csv('./data/temp/bj-data.csv')
    old_info = pd.read_csv('./data/raw/Beijing_aq_20180405.csv')
    old_info = pd.merge(old_info, new_info, how='outer')
    old_info.to_csv('./data/raw/Beijing_aq_20180405.csv', index=False)

    url = 'https://biendata.com/competition/airquality/ld/' + time_period + '/2k0d1d8'
    respones = requests.get(url)
    text = respones.text
    with open ("./data/temp/ld-data.csv", 'w') as f:
        f.write(text)
    new_info = pd.read_csv('./data/temp/ld-data.csv')
    old_info = pd.read_csv('./data/raw/London_aq_20180405.csv')
    old_info = pd.merge(old_info, new_info, how='outer')
    old_info.to_csv('./data/raw/London_aq_20180405.csv', index=False)

def UpdateMeoInfo():
    time_now = datetime.datetime.utcnow()
    time_text = time_now.strftime('20%y-%m-%d')
    start_time = time_text + '-0'
    end_time = time_text + '-23'
    time_period = start_time + '/' + end_time
    
    url = 'https://biendata.com/competition/meteorology/bj_grid/' + time_period + '/2k0d1d8'
    response = requests.get(url)
    with open('./data/temp/bj-data.csv', 'w') as f:
        f.write(response.text)
    new_info = pd.read_csv('./data/temp/bj-data.csv')
    old_info = pd.read_csv('./data/raw/Beijing_grid_20180405.csv')
    old_info = pd.merge(old_info, new_info, how='outer')
    old_info.to_csv('./data/raw/Beijing_grid_20180405.csv', index=False)    

    url = 'https://biendata.com/competition/meteorology/ld_grid/' + time_period + '/2k0d1d8'
    response = requests.get(url)
    with open('./data/temp/ld-data.csv', 'w') as f:
        f.write(response.text)
    new_info = pd.read_csv('./data/temp/ld-data.csv')
    old_info = pd.read_csv('./data/raw/London_grid_20180405.csv')
    old_info = pd.merge(old_info, new_info, how='outer')
    old_info.to_csv('./data/raw/London_grid_20180405.csv', index=False)    

# Rearrange aq data
UpdateAqData()
## load from files
data_bj = pd.read_csv('./data/raw/beijing_17_18_aq.csv')
data_bj['city'] = 'bj'

data_ld1 = pd.read_csv('./data/raw/London_historical_aqi_forecast_stations_20180331.csv')
data_ld1['city'] = 'ld'

data_ld2 = pd.read_csv('./data/raw/London_historical_aqi_other_stations_20180331.csv')
data_ld2['city'] = 'ld'

data_file = pd.concat([data_bj, data_ld1, data_ld2]) 

############# load from API ############
data_api_bj = pd.read_csv('./data/raw/Beijing_aq_20180405.csv')
data_api_bj['city'] = 'bj'
data_api_ld = pd.read_csv('./data/raw/London_aq_20180405.csv')
data_api_ld['city'] = 'ld'
data_api = pd.concat([data_api_bj, data_api_ld])
data_api = data_api.rename(index=str, columns={"time": "utc_time", "station_id": "stationId", "PM25_Concentration": "PM2.5",
                                              "PM10_Concentration": "PM10", "O3_Concentration": "O3", 
                                              "NO2_Concentration": "NO2", "SO2_Concentration": "SO2",
                                              "CO_Concentration": "CO"})
data_api = data_api.drop(columns=['id'])

############## concat ##############
data = pd.concat([data_file, data_api])
stations = set(data['stationId'])
data = data.loc[pd.notnull(data['utc_time']) & pd.notnull(data['stationId'])]
data['utc_time'] = pd.to_datetime(data['utc_time'])
data.drop_duplicates(subset=['stationId', 'utc_time'], inplace=True)
data['hour'] = data['utc_time'].dt.hour
data.reset_index(drop=True, inplace=True)

def check_no_dup(data):
    from collections import Counter 
    for s in stations:
        ind = data[data['stationId']==s].utc_time
        assert len(ind) == len(set(ind)), (s, Counter(ind))
check_no_dup(data)

gas = ['PM10', 'NO2', 'PM2.5', 'CO', 'O3', 'SO2']
def create_dataset(df, lagging=72):
    def fillna(df):
        df = df.copy()
        for c in gas:
            df[c] = df.groupby(['city', 'utc_time'], sort=False)[c].apply(lambda x: x.fillna(x.median()))
            df[c] = df.groupby(['stationId', 'hour'], sort=False)[c].apply(lambda x: x.fillna(x.median()))

        if df.isna().any().any():
            for c in gas:
                df[c] = df.groupby(['hour'], sort=False)[c].apply(lambda x: x.fillna(x.median()))
        return df
    
    df = fillna(df)
    return df

df = create_dataset(data)
assert not df.isna().any().any(), ("dataset still contains NaN.", df.isna().any())
df.to_csv('./data/temp/data_fillna.csv', index=False)
print('fillna done')

# Rearrange weather data

aq_info = pd.read_csv('./data/temp/data_fillna.csv')
aq_info.utc_time = pd.to_datetime(aq_info.utc_time)
match_info = pd.read_csv('./data/rearranged/aq_meo_match.csv')

aq_meo_match = {}
aq_grid_match = {}
match_cnt = match_info.shape[0]
for i in range(match_cnt):
    aq_meo_match[match_info['aq'][i]] = match_info['meo'][i]
    aq_grid_match[match_info['aq'][i]] = match_info['grid'][i]
print aq_info.head()

aq_info = aq_info[~aq_info['stationId'].isin(['BX9', 'BX1', 'CT2', 'CT3', 'CR8', 
                                             'GB0', 'HR1', 'LH0', 'KC1', 'RB7', 'TD5'])]
aq_info['meo'] = aq_info['stationId'].apply(lambda x:aq_meo_match[x])
aq_info['grid'] = aq_info['stationId'].apply(lambda x:aq_grid_match[x])

UpdateMeoInfo()

bj_meo_info = pd.read_csv('./data/raw/Beijing_historical_meo_grid.csv')
ld_meo_info = pd.read_csv('./data/raw/London_historical_meo_grid.csv')
bj_new_meo_info = pd.read_csv('./data/raw/Beijing_grid_20180405.csv')
ld_new_meo_info = pd.read_csv('./data/raw/London_grid_20180405.csv')

bj_meo_info = bj_meo_info.rename(index=str, columns={'stationName' : 'grid', 'wind_speed/kph' : 'wind_speed'})
ld_meo_info = ld_meo_info.rename(index=str, columns={'stationName' : 'grid', 'wind_speed/kph' : 'wind_speed'})
bj_new_meo_info = bj_new_meo_info.rename(index=str, columns={'station_id' : 'grid', 'time' : 'utc_time'})
ld_new_meo_info = ld_new_meo_info.rename(index=str, columns={'station_id' : 'grid', 'time' : 'utc_time'})
bj_new_meo_info = bj_new_meo_info.drop(['id'], axis=1)
ld_new_meo_info = ld_new_meo_info.drop(['id'], axis=1)

meo_info = pd.concat([bj_meo_info, ld_meo_info, bj_new_meo_info, ld_new_meo_info])

aq_info = pd.merge(aq_info, meo_info, on=['grid', 'utc_time'], how='left')
aq_info.to_csv('./data/data.csv', index=False)
visAll(aq_info)
visAll(data, 'has_na')