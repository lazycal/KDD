#encoding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from collections import Counter 
# import xgboost as xgb
import datetime
from vis import visAll

#encoding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from collections import Counter 
# import xgboost as xgb
import datetime

def fetch_forecast_city(city):
    time_now = datetime.datetime.utcnow()
    cnt = 0
    fetched = False
    while not fetched and cnt < 48:
        time_text = time_now.strftime('20%y-%m-%d-%H')
        start_time = time_text
        print(time_text)

        url = 'http://kdd.caiyunapp.com/competition/forecast/{}/'.format(city) + start_time + '/2k0d1d8'
        respones = requests.get(url)
        text = respones.text
        if text == 'None':
            print('forecast response=', text)
            cnt += 1
            time_now -= datetime.timedelta(hours=1)
            continue
        fetched = True
        with open ("./data/temp/{}-forecast.csv".format(city), 'w') as f:
            f.write(text)
        return pd.read_csv('./data/temp/{}-forecast.csv'.format(city), parse_dates=['forecast_time'])

def fetch_forecast():
    bj = fetch_forecast_city('bj')
    ld = fetch_forecast_city('ld')
    if bj is not None and ld is not None:
        df = pd.concat([bj, ld]).drop_duplicates(keep='last', subset=['forecast_time', 'station_id'])
        df = df.rename(index=str, columns={'station_id' : 'grid', 'forecast_time': 'utc_time'}).drop(['id'], axis=1)
        df = df.sort_values(['grid', 'utc_time'])
        # fillna
        if df.isna().any().any():
            for c in (set(df.columns) - set(['utc_time', 'grid'])):
                if df[c].isna().any():
                    df[c] = df[c].fillna(method='ffill')
        df.to_csv('./data/forecast.csv', index=False)

def UpdateAqData():
    time_now = datetime.datetime.utcnow()
    time_text = time_now.strftime('20%y-%m-%d')
    start_time = (time_now + pd.DateOffset(n=-2)).strftime('20%y-%m-%d') + '-0'
    end_time = time_text + '-23'
    time_period = start_time + '/' + end_time

    url = 'https://biendata.com/competition/airquality/bj/' + time_period + '/2k0d1d8'
    respones = requests.get(url)
    text = respones.text
    with open ("./data/temp/bj-data.csv", 'w') as f:
        f.write(text)
    new_info = pd.read_csv('./data/temp/bj-data.csv', parse_dates=['time'])
    old_info = pd.read_csv('./data/raw/Beijing_aq_20180405.csv', parse_dates=['time'])
    old_info = pd.concat([old_info, new_info]).drop_duplicates(keep='last', subset=['time', 'station_id'])
    old_info.sort_values(by=['time']).to_csv('./data/raw/Beijing_aq_20180405.csv', index=False)

    url = 'https://biendata.com/competition/airquality/ld/' + time_period + '/2k0d1d8'
    respones = requests.get(url)
    text = respones.text
    with open ("./data/temp/ld-data.csv", 'w') as f:
        f.write(text)
    new_info = pd.read_csv('./data/temp/ld-data.csv', parse_dates=['time'])
    old_info = pd.read_csv('./data/raw/London_aq_20180405.csv', parse_dates=['time'])
    old_info = pd.concat([old_info, new_info]).drop_duplicates(keep='last', subset=['time', 'station_id'])
    old_info.sort_values(by=['time']).to_csv('./data/raw/London_aq_20180405.csv', index=False)

def UpdateMeoInfo():
    time_now = datetime.datetime.utcnow()
    time_text = time_now.strftime('20%y-%m-%d')
    start_time = (time_now + pd.DateOffset(n=-2)).strftime('20%y-%m-%d') + '-0'
    end_time = time_text + '-23'
    time_period = start_time + '/' + end_time
    
    url = 'https://biendata.com/competition/meteorology/bj_grid/' + time_period + '/2k0d1d8'
    response = requests.get(url)
    with open('./data/temp/bj-data.csv', 'w') as f:
        f.write(response.text)
    new_info = pd.read_csv('./data/temp/bj-data.csv', parse_dates=['time'])
    old_info = pd.read_csv('./data/raw/Beijing_grid_20180405.csv', parse_dates=['time'])
    old_info = pd.concat([old_info, new_info]).drop_duplicates(keep='last', subset=['time', 'station_id'])
    old_info.sort_values(by=['time']).to_csv('./data/raw/Beijing_grid_20180405.csv', index=False)    

    url = 'https://biendata.com/competition/meteorology/ld_grid/' + time_period + '/2k0d1d8'
    response = requests.get(url)
    with open('./data/temp/ld-data.csv', 'w') as f:
        f.write(response.text)
    new_info = pd.read_csv('./data/temp/ld-data.csv', parse_dates=['time'])
    old_info = pd.read_csv('./data/raw/London_grid_20180405.csv', parse_dates=['time'])
    old_info = pd.concat([old_info, new_info]).drop_duplicates(keep='last', subset=['time', 'station_id'])
    old_info.sort_values(by=['time']).to_csv('./data/raw/London_grid_20180405.csv', index=False)    

fetch_forecast()
print('forecast fetched')
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

############# fill blank hours #####
bj_cities = set(data[data.city == 'bj'].stationId.unique())
future60 = datetime.datetime.utcnow() + datetime.timedelta(hours=60)
date_range = pd.date_range('2017-03-01 00:00:00', future60, freq='1H')
new_index = pd.MultiIndex.from_product([data['stationId'].unique(), date_range],
                                       names=['stationId', 'utc_time'])
df1 = pd.DataFrame(index=new_index).reset_index()
data = pd.merge(df1, data, on=['stationId', 'utc_time'], how='left', validate='1:m')
data.city = data.stationId.map(lambda x: 'bj' if x in bj_cities else 'ld')
data = data.sort_values(by=['utc_time', 'stationId'])

############# add hour column #####
data['hour'] = data['utc_time'].dt.hour
data.reset_index(drop=True, inplace=True)

############ check no duplicates
def check_no_dup(data):
    for s in stations:
        ind = data[data['stationId']==s].utc_time
        assert len(ind) == len(set(ind)), (s, Counter(ind))
check_no_dup(data)
data.to_csv('./data/temp/phase1.csv', index=False)
print("Finish phase1")
# Rearrange weather data

# aq_info = pd.read_csv('./data/temp/data_fillna.csv')
aq_info = data
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

bj_meo_info = pd.read_csv('./data/raw/Beijing_historical_meo_grid.csv', parse_dates=['utc_time'])
ld_meo_info = pd.read_csv('./data/raw/London_historical_meo_grid.csv', parse_dates=['utc_time'])
bj_new_meo_info = pd.read_csv('./data/raw/Beijing_grid_20180405.csv', parse_dates=['time'])
ld_new_meo_info = pd.read_csv('./data/raw/London_grid_20180405.csv', parse_dates=['time'])
forc_meo_info = pd.read_csv('./data/forecast.csv', parse_dates=['utc_time'])

bj_meo_info = bj_meo_info.rename(index=str, columns={'stationName' : 'grid', 'wind_speed/kph' : 'wind_speed'})
ld_meo_info = ld_meo_info.rename(index=str, columns={'stationName' : 'grid', 'wind_speed/kph' : 'wind_speed'})
bj_new_meo_info = bj_new_meo_info.rename(index=str, columns={'station_id' : 'grid', 'time' : 'utc_time'})
ld_new_meo_info = ld_new_meo_info.rename(index=str, columns={'station_id' : 'grid', 'time' : 'utc_time'})
bj_new_meo_info = bj_new_meo_info.drop(['id'], axis=1)
ld_new_meo_info = ld_new_meo_info.drop(['id'], axis=1)

meo_info = pd.concat([bj_meo_info, ld_meo_info, bj_new_meo_info, ld_new_meo_info, forc_meo_info])
meo_info.utc_time = pd.to_datetime(meo_info.utc_time)

aq_info = pd.merge(aq_info, meo_info, on=['grid', 'utc_time'], how='left')
bj_wea_info = pd.concat([
    pd.read_csv('./data/raw/beijing_17_18_meo.csv', parse_dates=['utc_time']), 
    pd.read_csv('./data/raw/beijing_201802_201803_me.csv', parse_dates=['utc_time'])
]).rename(index=str, columns={'station_id' : 'meo'}).drop_duplicates(subset=['utc_time', 'meo'])

aq_info = aq_info.set_index(['utc_time', 'meo'])
bj_wea_info = bj_wea_info.set_index(['utc_time', 'meo'])[['weather']]

aq_info.update(bj_wea_info)
aq_info = aq_info.reset_index()

aq_info = aq_info.drop_duplicates(subset=['utc_time', 'stationId']) ## important
# assert aq_info.duplicated(subset=['utc_time', 'stationId']).any() == False, aq_info.duplicated(subset=['utc_time', 'stationId'])
aq_info = aq_info.sort_values(by=['utc_time', 'stationId'])
aq_info.to_csv('./data/data_na.csv', index=False)

############# fillna ###################
print('performing fillna')

def fillna(df):
    # gas = set(df.columns)-set(['utc_time', 'city', 'stationId', 'meo', 'grid', 'weather']) # TODO: Add weather 
    gas = set(['PM10', 'NO2', 'PM2.5', 'CO', 'O3', 'SO2']) # TODO: more collumns
    df = df.copy()
    def fill(df, group):
        a = df.groupby(group, sort=False)
        for c in gas:
            if df[c].isna().any():
                df[c] = a[c].apply(lambda x: x.fillna(x.median()))
    fill(df, ['city', 'utc_time'])
   
    df = df.sort_values(['stationId', 'utc_time'])
    for c in (set(df.columns) - gas - set(['utc_time', 'city', 'stationId', 'meo', 'grid'])):
        if df[c].isna().any():
            df[c] = df[c].fillna(method='ffill')

    # if df.isna().any().any():
    #     fill(df, ['stationId', 'hour'])
        
#     if df.isna().any().any():
#         for c in gas:
#             df[c] = df.groupby(['hour'], sort=False)[c].apply(lambda x: x.fillna(x.median()))
# TODO: drop
    return df

aq_info_fillna = fillna(aq_info).sort_values(['utc_time', 'stationId'])
# assert not aq_info_fillna.isna().any().any(), ("dataset still contains NaN.", aq_info_fillna.isna().any())
# aq_info.to_csv('./data/temp/data_fillna.csv', index=False)
print('fillna done')
aq_info_fillna.to_csv('./data/data.csv', index=False)
# visAll(aq_info_fillna)
# visAll(aq_info, 'has_na')