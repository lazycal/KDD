# coding: utf-8
# pylint: disable = invalid-name, C0111
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import dataset
from dataset import lag_format
import matplotlib.pyplot as plt
import os
import sys
import time
import datetime
from datetime import timedelta
from myprofile import *
from collections import Counter
import re
import argparse

def createGenNext(self, target, df=None, deploy=False): # df: initial value
    # @profile
    def find_new_X_test(new_X_test):
        tmp = df[df.utc_time == new_X_test.iloc[0].utc_time]
        # print('len(tmp)={}'.format(len(tmp)))
        if len(tmp) > 0:
            assert len(tmp) == 1
            return tmp, True
        return None, False

    # @profile
    def genNext(X_test, y_pred, features):
        assert len(X_test) == 1
        new_X_test = X_test.copy()
        new_X_test.utc_time += pd.DateOffset(hours=1)
        if 'hour' in new_X_test.columns:
            new_X_test['hour'] = (1 + new_X_test['hour']) % 24
        find = False
        if df is not None: 
            tmp, find = find_new_X_test(new_X_test)
            assert find, 'date {} not found'.format(new_X_test.utc_time)
            tmp = tmp.copy().reset_index(drop=True)

        new_X_test = new_X_test.reset_index(drop=True)
        for c in ([target] + self.lag_cols):
            for i in range(1, self.lags):
                new_X_test[lag_format(c, i + 1)] = new_X_test[lag_format(c, i)]
            new_X_test[lag_format(c, 1)] = new_X_test[c]
            if df is not None:
                new_X_test[c] = tmp[c]
        if not deploy:
            new_X_test[lag_format(target, 1)] = y_pred
        return new_X_test
    return genNext

def createGenNextOld(target, df=None, deploy=False): # df: initial value
    # @profile
    def find_new_X_test(new_X_test):
        tmp = df[df.utc_time == new_X_test.iloc[0].utc_time]
        # print('len(tmp)={}'.format(len(tmp)))
        if len(tmp) > 0:
            assert len(tmp) == 1
            return tmp, True
        return None, False

    # @profile
    def genNext(X_test, y_pred, features):
        assert len(X_test) == 1
        new_X_test = X_test.copy()
        new_X_test.utc_time += pd.DateOffset(hours=1)
        if 'hour' in new_X_test.columns:
            new_X_test['hour'] = (1 + new_X_test['hour']) % 24
        find = False
        if df is not None: 
            tmp, find = find_new_X_test(new_X_test)
            if find:
                new_X_test = tmp.copy()
        # lags = [_ for _ in X_test.column if _.find(lag_format(target, '')) != -1] 
        new_X_test = new_X_test.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        for c in features:
            match = re.match('(.*?)_lag_(\d+)', c)
            if match is not None:
                name = match.group(1)
                lag_idx = int(match.group(2))
                if lag_idx > 1:
                    new_X_test[c] = X_test['{}_lag_{}'.format(name, lag_idx - 1)]
                elif name in X_test.columns and name != target:
                    new_X_test[c] = X_test[name]
        if not find or not deploy or new_X_test[lag_format(target, 1)].isna().any():
            new_X_test[lag_format(target, 1)] = y_pred
        return new_X_test
    return genNext

# @profile
def genFeatures2(t, df): # arma
    cols = [t]
    features = []
    for c in cols:
        features += [_ for _ in df.columns if _.find(lag_format(c, '')) != -1]
    features += ['hour']
    return features

class Strategy(object):
    def __init__(self, args):
        self.args = args
        self.dataset = dataset.create_dataset1('../data/data.csv', MaxLagging=args.lag)

    def setStation(self, stationId):
        self.stationId = stationId
        self.cur_station_df = self.dataset[self.dataset.stationId == stationId]

    def genFeatures(self):
        pass

    def createGenNext(self, target, deploy=False):
        return createGenNextOld(target)
    
    def gbm_predict(self, gbm, df, length, st, genNext, features): # No nan allowed in df[features]
        assert not df[features].isna().any().any(), 'gbm_predict: No nan allowed in df[features]'
        X_test = df[df.utc_time <= st].iloc[-1:]
        # X_test = X_test[features]
        ed = st + datetime.timedelta(hours=length)
        y_pred = []
        t = X_test.iloc[0].utc_time
        print('To predict {}, start from {}'.format(st, t))
        while t != ed:
            # print(X_test[['utc_time']+features])
            y_pred.append(max(0, gbm.predict(X_test[features])[0]))
            X_test = genNext(X_test, y_pred[-1], features)
            t += datetime.timedelta(hours=1)

        print(y_pred)
        return y_pred[-length:]

class Strategy1(Strategy):
    def __init__(self, args):
        Strategy.__init__(self, args)
        self.lag_cols = ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed', 'weather']
        self.lags = args.lag
        print('Strategy1: \nlag: {}, using: {}\n'.format(args.lag, self.lag_cols + ['hour']))
        # self.forc = pd.read_csv('../data/forcast.csv', parse_dates=['utc_time'])
        # self.forc.weather = self.forc.weather.astype('category')

        # match_info = pd.read_csv('./data/rearranged/aq_meo_match.csv')
        # aq_grid_match = {}
        # match_cnt = match_info.shape[0]
        # for i in range(match_cnt):
        #     aq_grid_match[match_info['aq'][i]] = match_info['grid'][i]
        # self.aq_grid_match = aq_grid_match

        # data_range = pd.date_range(self.forc.utc_time.min(), self.forc.utc_time.max())
        # self.df = pd.merge(self.dataset, self.forc, on=['grid', 'utc_time'], how='left')

        # match_info = pd.read_csv('../data/rearranged/aq_meo_match.csv')
        # aq_grid_match = {}
        # match_cnt = match_info.shape[0]
        # for i in range(match_cnt):
        #     aq_grid_match[match_info['aq'][i]] = match_info['grid'][i]

        # date_range = pd.date_range(self.df.utc_time.min(), self.df.utc_time.max())
        # self.df = pd.concat([self.dataset, self.df]).drop_duplicates(keep='first', subset=['utc_time', 'stationId'])
        # self.df.to_csv("../data/temp/test.csv")
        
    def genFeatures(self, t, df, **kwargs):
        cols = self.lag_cols + [t]
        if df['weather'].isna().all():
            print('London city')
            cols.remove('weather')
        features = []
        for c in cols:
            features += [_ for _ in df.columns if _.find(lag_format(c, '')) != -1]
        features += ['hour'] + cols#, 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed']
        features.remove(t)
        return features   

    def createGenNext(self, target, deploy=False):
        # return createGenNext(self, target, self.cur_station_df, deploy=deploy)
        return createGenNextOld(target, self.cur_station_df, deploy=deploy)

class Strategy2(Strategy):
    def __init__(self, args):
        Strategy.__init__(self, args)
        print('Strategy2: using: arma\n')
        
    def genFeatures(self, t, df, **kwargs): #
        return genFeatures2(t, df)

    def createGenNext(self, target, deploy=False):
        return createGenNextOld(target, self.cur_station_df, deploy=deploy)

def getStrategy(name, args):
    if name == 'Strategy1':
        return Strategy1(args)
    elif name == 'Strategy2':
        return Strategy2(args)
    else:
        raise ValueError("No such strategy")

# PM2.5 wind_speed hour wind_dir weather humidity
# PM10  hour wind_speed humidity wind_dir
# O3:   hour temp wind_speed humidity wind_dir


def _gen_from_dict(d):
    res = []
    for k, v in d.items():
        if type(v) == int:
            v = range(v + 1)
        for i in v:
            res.append(lag_format(k, i))
    return res

class Strategy3(Strategy):
    def __init__(self, args):
        Strategy.__init__(self, args)
        self.PM25 = {
            'wind_speed': 3,
            'wind_direction': 5,
            'PM2.5': args.lag,
            'humidity': 0,
            'weather': 0,
            'hour': 0,
        }
        self.PM10 = {
            'wind_speed': [0, 1, 2, 13],
            'wind_direction': 5,
            'PM10': args.lag,
            'humidity': [0, 4, 5, 24],
            'weather': 0,
            'pressure': 0,
            'hour': 0
        }
        self.O3 = {
            'O3': args.lag,
            'hour': 0,
            'wind_speed': 4,
            'temperature': 1,
            'humidity': 0,
            'wind_direction': 1,
        }
        self.features = {
            'PM2.5': _gen_from_dict(self.PM25),
            'PM10': _gen_from_dict(self.PM10),
            'O3': _gen_from_dict(self.O3)
        }
        print('Strategy3: using: PM2.5\n')
        
    def genFeatures(self, t, df, **kwargs): #
        features = []
        for c in self.features[t]:
            if c in df.columns:
                features.append(c)
        features.remove(t)
        print(features)
        return features

    def createGenNext(self, target, deploy=False):
        return createGenNextOld(target, self.cur_station_df, deploy=deploy)


class Strategy4(Strategy):
    def __init__(self, args):
        Strategy.__init__(self, args)
        self.features = pd.read_csv("../data/import_table.csv")
        self.thres = args.thres
        print('Strategy3: using import_table\n')
        
    def genFeatures(self, t, df, **kwargs): #
        s = self.stationId
        features = list(self.features[(self.features.stationId == s) & (self.features.gas == t) & (self.features.importance > self.thres)].feature)
        self_lag = []
        for f in features:
            match = re.match(r'{}_lag_(\d+)'.format(t), f)
            if match is not None:
                self_lag.append(int(match.group(1)))
        self_lag = set(self_lag)
        for i in range(1, max(self_lag)):
            if i not in self_lag:
                features.append(lag_format(t, i))
        print(features)
        return features

    def createGenNext(self, target, deploy=False):
        return createGenNextOld(target, self.cur_station_df, deploy=deploy)


def getStrategy(name, args):
    if name == 'Strategy1':
        return Strategy1(args)
    elif name == 'Strategy2':
        return Strategy2(args)
    elif name == 'Strategy3':
        return Strategy3(args)
    elif name == 'Strategy4':
        return Strategy4(args)
    else:
        raise ValueError("No such strategy")