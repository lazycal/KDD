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

def createGenNext(target, df=None, deploy=False): # df: initial value
    def genNext(X_test, y_pred, features):
        assert len(X_test) == 1
        new_X_test = X_test.copy()
        new_X_test.utc_time += pd.DateOffset(hours=1)
        find = False
        if df is not None: 
            tmp = df[df.utc_time == new_X_test.iloc[0].utc_time]
            # print('len(tmp)={}'.format(len(tmp)))
            if len(tmp) > 0:
                assert len(tmp) == 1
                new_X_test = tmp.copy()
                find = True
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
        if 'hour' in new_X_test.columns:
            new_X_test['hour'] = (1 + new_X_test['hour']) % 24
        if not find or not deploy or new_X_test[lag_format(target, 1)].isna().any():
            new_X_test[lag_format(target, 1)] = y_pred
        return new_X_test
    return genNext

# @profile
def genFeatures1(t, df):
    cols = ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed', 'weather', t]
    if df['weather'].isna().all():
        print('London city')
        cols.remove('weather')
    features = []
    for c in cols:
        features += [_ for _ in df.columns if _.find(lag_format(c, '')) != -1]
    features += ['hour'] + cols#, 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed']
    features.remove(t)
    return features   

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
        return createGenNext(target)

class Strategy1(Strategy):
    def __init__(self, args):
        Strategy.__init__(self, args)
        print('Strategy1: using: \n')
        print(['hour', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed'])
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
        
    def genFeatures(self, t, df):
        return genFeatures1(t, df)

    def createGenNext(self, target, deploy=False):
        return createGenNext(target, self.cur_station_df, deploy=deploy)

class Strategy2(Strategy):
    def __init__(self, args):
        Strategy.__init__(self, args)
        print('Strategy2: using: arma\n')
        
    def genFeatures(self, t, df): #
        return genFeatures2(t, df)

    def createGenNext(self, target, deploy=False):
        return createGenNext(target, self.cur_station_df, deploy=deploy)

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