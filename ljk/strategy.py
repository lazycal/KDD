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
from numba import jit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
import copy
import ast
# @profile
def _smape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    # actual = map(lambda x: np.nan if x < 0 else x, actual)
    # predicted = map(lambda x: np.nan if x < 0 else x, predicted)
    actual = np.maximum(actual, 0)
    predicted = np.maximum(predicted, 0)
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)
    return float(2 * np.nanmean(np.divide(a, b, out=np.zeros_like(a), where=(b!=0), casting='unsafe')))

def _smape1(actual, predicted):
    return "smape", _smape(actual, predicted), False
# @profile
def smape(predicted, train_data):
    actual = train_data.get_label()
    return 'smape', _smape(actual, predicted), False

def createGenNextOld(target, df=None, deploy=False): # df: initial value
    # @profile
    @jit
    def find_new_X_test(new_X_test):
        tmp = df[df.utc_time == new_X_test.iloc[0].utc_time]
        # print('len(tmp)={}'.format(len(tmp)))
        if len(tmp) > 0:
            assert len(tmp) == 1
            return tmp, True
        return None, False

    # @profile
    @jit
    def genNext(X_test, y_pred, features, ref):
        assert len(X_test) == 1
        assert ref is not None

        ##### SLOW ######
        # new_X_test = X_test.copy()
        # new_X_test.utc_time += pd.DateOffset(hours=1)
        # if 'hour' in new_X_test.columns:
        #     new_X_test['hour'] = (1 + new_X_test['hour']) % 24
        # find = False
        # if df is not None: 
        #     tmp, find = find_new_X_test(new_X_test)
        #     if find:
        #         new_X_test = tmp.copy()
        ##### SHOULD BE LESS SLOW ######
        find = True
        new_X_test = ref

        new_X_test = new_X_test.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        for c in features:
            match = re.match('(.*?)_lag_(\d+)', c)
            if match is not None:
                name = match.group(1)
                lag_idx = int(match.group(2))
                if lag_idx > 1:
                    col = '{}_lag_{}'.format(name, lag_idx - 1)
                    if col in X_test.columns:
                        new_X_test[c] = X_test[col]
                elif name in X_test.columns and name != target:
                    new_X_test[c] = X_test[name]
        if not find or not deploy or new_X_test[lag_format(target, 1)].isna().any():
            new_X_test[lag_format(target, 1)] = y_pred
        # print(np.expm1(new_X_test[ [lag_format(target, i) for i in range(13)] ].iloc[0].values))
        return new_X_test
    return genNext

# @profile
def genFeatures2(t, df): # arma
    cols = [t]
    features = []
    for c in cols:
        features += [_ for _ in df.columns if _.find(lag_kformat(c, '')) != -1]
    features += ['hour']
    return features

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

class Strategy(object):
    def __init__(self, args, creator=dataset.create_dataset1):
        self.args = args
        self.dataset = creator(args.data_path, MaxLagging=args.lag)
        if args.predict_date is not None:
            for gas in ['PM2.5', 'PM10', 'O3']:
                for i in range(args.lag + 1):
                    self.dataset.loc[self.dataset.utc_time >= args.predict_date, lag_format(gas, i)] = np.nan
                    # print('Latest one with non-nan {} in dataset : '.format(lag_format(gas, i)), self.dataset.dropna(subset=[lag_format(gas, i)]).utc_time.max())
        print('Latest of dataset: ', self.dataset.utc_time.max())

    def setStation(self, stationId):
        self.stationId = stationId
        self.cur_station_df = self.dataset[self.dataset.stationId == stationId]

    def genFeatures(self, t, df, **kwargs):
        ld = self.stationId[0].isupper()
        self.ld = ld # thread-unsafe
        self.city = 'ld' if self.ld else 'bj'
        self.target = t # thread-unsafe

    def getW(self, df):
        return None

    def split(self, df, ratio):
        pivot = pd.to_datetime('2018-05-01')
        # df_test = df_test.loc[df_test['utc_time'] < pd.to_datetime('2018-05-18')] # use the same dataset as validation set
        # df_train = df.loc[df['utc_time'] < pd.to_datetime('2018-04-01')]
        # df_test = df.loc[df['utc_time'] >= pd.to_datetime('2018-04-01')]
        df_train = df.loc[df['utc_time'] < pivot]
        df_test = df.loc[df['utc_time'] >= pivot]
        if not self.args.deploy:
        #     # df_test = df.loc[df['utc_time'] >= pd.to_datetime('2018-05-01')]
            df_test = df_test.loc[df_test['utc_time'] < pd.to_datetime('2018-05-21')] # use the same dataset as validation set
            # df_test = df_test.loc[df_test['utc_time'] < pd.to_datetime('2018-05-04')] # use the same dataset as validation set

        return df_train, df_test
    def getParams(self, *args, **kwargs):
        args = self.args
        return [{
            # 'max_bin': args.max_bin,
            'num_threads': args.num_threads,
            'boosting_type': args.boosting,
            'objective': args.objective,
            'min_data_in_leaf': args.min_data_in_leaf,
            # 'metric': 'na',
            'num_leaves': args.num_leaves,
            'learning_rate': args.lr,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }]

    def train(self, df, features, target, ratio):
        args = self.args
        assert(ratio[1] == 'num')
        print('target=', target)
        df = df.dropna(subset=features + [target]) # TODO: dropna consecutive violated
        df_train, df_test = self.split(df, ratio)
        # df_train, df_test = df.iloc[:-ratio[0]], df.iloc[-ratio[0]:]
        assert df_train.utc_time.max() < df_test.utc_time.min(), (df_train.utc_time.max(), df_test.utc_time.min())
        date_range = df_test.utc_time

        print(df_train.tail())
        print('val date_range={}'.format(date_range))

        y_train = df_train[target]
        y_test = df_test[target]
        X_train = df_train[features]
        X_test = df_test[features]
        W_train = self.getW(df_train)
        W_test = self.getW(df_test)

        num_train, num_feature = X_train.shape
        print('num_train={}, num_feature={} num_test={}'.format(num_train, num_feature, len(X_test)))

        # create dataset for lightgbm
        # if you want to re-use data, remember to set free_raw_data=False
        lgb_train = lgb.Dataset(X_train, y_train,
                                weight=W_train, free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                            weight=W_test, free_raw_data=False)

        # specify your configurations as a dict
        params = copy.deepcopy(self.getParams(df, features, target))
        best_score = 1e10
        best_model = None
        for idx, param in enumerate(params):
            print('#{} param='.format(idx))
            print(param)
            print('Start training...')

            gbm = lgb.train(param,
                            lgb_train,
                            num_boost_round=args.max_iter,
                            valid_sets=[lgb_eval, lgb_train],
                            feval=smape,
                            valid_names=['eval', 'train'],
                            early_stopping_rounds=args.es if args.es > 0 else None)
            print('done')
            score = _smape(gbm.predict(lgb_eval.data), lgb_eval.label)
            print('Finish fitting. Score is {}'.format(score))
            if score < best_score:
                best_score = score
                best_model = gbm
        gbm = best_model
        print('best score = {}'.format(best_score))
        # feature importances
        importance = list(100 * gbm.feature_importance().astype(np.float) / gbm.feature_importance().sum())
        print('Feature importances:')
        importance = pd.DataFrame({"feature": gbm.feature_name(), "importance": importance})
        print(importance.sort_values(by=['importance']))

        return gbm, np.nan, date_range, importance

    def createGenNext(self, target, deploy=False):
        return createGenNextOld(target)

    def gbm_predict(self, gbm, df, length, st, genNext, features): # No time gaps allowed in df
        # assert not df[features].isna().any().any(), 'gbm_predict: No nan allowed in df[features]'
        df = self.cur_station_df  # dirty and won't be fixed
        df_drop = df[features + ['utc_time']].dropna()
        X_test = df_drop[df_drop.utc_time <= st].iloc[-1:]
        ed = st + datetime.timedelta(hours=length)
        y_pred = []
        t = X_test.iloc[0].utc_time
        df = df[df.utc_time > t]
        idx = 0
        print('To predict {}, start from {}'.format(st, t))
        if self.args.debug:
            delta = None
        while t != ed:
            # print(X_test[['utc_time']+features])
            y_pred.append(max(0, gbm.predict(X_test[features])[0]))
            ######## test begin ######
            if self.args.debug:
                def test():
                    print('test')
                    tmp = X_test[features].copy()
                    tmp1 = []
                    x = range(0, 100, 10)
                    for i in x:
                        tmp.rain_hours = i
                        tmp.last_rain = i
                        tmp1.append(gbm.predict(tmp)[0])
                    tmp1 = np.array(tmp1) - tmp1[0]
                    print(tmp1)
                    return tmp1
                if delta is None:
                    delta = test()
                else:
                    delta += test()
            ########### test end ######
            X_test = genNext(X_test, y_pred[-1], features, df.iloc[idx: idx+1])
            t += datetime.timedelta(hours=1)
            assert t == df.iloc[idx].utc_time, df.iloc[idx].utc_time
            idx += 1

        if self.args.debug:
            x = range(0, 100, 10)
            plt.plot(x, delta / 48)
            plt.title(self.target)
            plt.show()
        print(y_pred)
        return y_pred[-length:]

class Strategy1(Strategy):
    def __init__(self, args, *other):
        Strategy.__init__(self, args, *other)
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
        Strategy.genFeatures(self, t, df, **kwargs)
        cols = self.lag_cols + [t]
        if self.stationId[0].isupper():
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
class Strategy5(Strategy1):
    def __init__(self, args):
        Strategy1.__init__(self, args, dataset.create_dataset5)
        print("Strategy5 log1p")
        
    def gbm_predict(self, *args, **kwargs):
        res = Strategy1.gbm_predict(self, *args, **kwargs)
        res = list(np.expm1(res))
        print(res)
        return res
class Strategy6(Strategy):
    def __init__(self, args, *other):
        Strategy.__init__(self, args, *other)
        self.lag_cols = ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed', 'weather_clu']
        self.lags = args.lag
        print('Strategy6: \nlag: {}, using: {}\n'.format(args.lag, self.lag_cols + ['hour', 'last_rain']))

        aq_info = self.dataset
        aq_info = aq_info.sort_values(by=['stationId', 'utc_time']).reset_index(drop='True')
        rind = aq_info[aq_info.weather_clu == '1'].index
        rind=rind.union(pd.Index([-100]))
        aq_info['last_rain']=aq_info.index.map(lambda x: rind[rind.get_loc(x, method='ffill')])
        aq_info.last_rain = np.minimum(aq_info.index - aq_info.last_rain, 100)
        aq_info = aq_info.sort_values(by=['utc_time', 'stationId']).reset_index(drop=True)
        # aq_info = aq_info[aq_info.utc_time >= '2017-04-01']
        self.dataset = aq_info
        # assert aq_info.last_rain.isna().any() == False, aq_info[aq_info.last_rain.isna()]
        print(aq_info[aq_info.utc_time <= '2018-05-15'].tail())
        print(aq_info[['last_rain', 'utc_time']].tail())
        print('aq_info[\'last_rain\'].mean()={}'.format(aq_info['last_rain'].mean()))
    
    def split(self, df, ratio):
        df_train, df_test = Strategy.split(self, df, ratio)
        if self.ld is False and self.target != 'O3':
            print('>=2017-04-01')
            df_train = df_train[df_train.utc_time >= '2017-04-01']
        return df_train, df_test

    def genFeatures(self, t, df, **kwargs):
        cols = self.lag_cols + [t]
        ld = self.stationId[0].isupper()
        self.ld = ld # thread-unsafe
        self.target = t # thread-unsafe
        if ld is True:
            print('London city')
            cols.remove('weather_clu')
        features = []
        for c in cols:
            features += [_ for _ in df.columns if _.find(lag_format(c, '')) != -1]
        features += ['hour'] + cols#, 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed']
        if ld is False and self.target != 'O3':
            features.append('last_rain')
        features.remove(t)

        if self.target == 'O3':
            res1 = []
            for f in features:
                if f.startswith('weather_clu'):
                    res1.append('weather'+f[len('weather_clu'):])
                else:
                    res1.append(f)
            features = res1
        return features   

    def createGenNext(self, target, deploy=False):
        # return createGenNext(self, target, self.cur_station_df, deploy=deploy)
        func = createGenNextOld(target, self.cur_station_df, deploy=deploy)
        station = self.stationId
        def genNext(*args, **kwargs):
            new_X_test = func(*args, **kwargs)
            if station.isupper() is False: # bj
                if new_X_test.weather_clu.iloc[0] == '1':
                    new_X_test.last_rain = 0
                else:
                    new_X_test.last_rain += 1
            return new_X_test
        return genNext
    
    def getW(self, df):
        # thread-unsafe
        if self.ld is True or self.target == 'O3':
            return None
        print('getW reutrn W')
        return df['last_rain'].map(lambda x: 1. / (x + 1))
class Strategy7(Strategy6):
    def getW(self, df):
        res = ((df['PM2.5']-df['PM2.5_lag_1']).map(abs) + (df['PM10']-df['PM10_lag_1']).map(abs) + 0.1).fillna(0.1)
        return res
class Strategy8(Strategy6):
    def genFeatures(self, *args, **kwargs):
        res = Strategy6.genFeatures(self, *args, **kwargs)
        res1 = []
        for f in res:
            if f.startswith('weather_clu'):
                res1.append('weather'+f[len('weather_clu'):])
            else:
                res1.append(f)
        return res1
class Strategy9(Strategy6):
    def getW(self, df):
        num_rain = len(df[df['last_rain'] < 10])
        tot = len(df)
        rr = float(num_rain) / tot
        ro = 1 - rr
        print('rr={}, ro={}'.format(rr, ro))
        return df['last_rain'].map(lambda x: ro if x < 10 else rr)
class Strategy10(Strategy):
    def func(self, x):
        n = len(x)
        w = self.DECAY**(np.linspace(0, n - 1, num=n))
        return np.dot(w, x)

    def __init__(self, args, *other):
        Strategy.__init__(self, args, *other)
        self.lag_cols = ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed', 'weather_clu']
        self.lags = args.lag
        self.WIN = 8
        self.DECAY = 1.5
        print('Strategy10: \nlag: {}, using: {}\n'.format(args.lag, self.lag_cols + ['hour', 'rain_hours']))

        aq_info = self.dataset
        aq_info = aq_info.sort_values(by=['stationId', 'utc_time']).reset_index(drop='True')
        aq_info['rain_hours'] = (aq_info.weather_clu == '1').rolling(window=self.WIN).agg(self.func)
        aq_info = aq_info.sort_values(by=['utc_time', 'stationId']).reset_index(drop=True)
        # aq_info = aq_info[aq_info.utc_time >= '2017-04-01']
        self.dataset = aq_info
        # assert aq_info.rain_hours.isna().any() == False, aq_info[aq_info.rain_hours.isna()]
        print(aq_info[aq_info.utc_time <= '2018-05-15'].tail())
        print(aq_info[['rain_hours', 'utc_time']].tail())
        print('aq_info[\'rain_hours\'].mean()={}'.format(aq_info['rain_hours'].mean()))
        print('aq_info[\'rain_hours\'].max()={}'.format(aq_info['rain_hours'].max()))
        print('aq_info[\'rain_hours\'].min()={}'.format(aq_info['rain_hours'].min()))

    def split(self, df, ratio):
        df_train, df_test = Strategy.split(self, df, ratio)
        if self.ld is False:
            df_train = df_train[df_train.utc_time >= '2017-04-01']
        return df_train, df_test

    def genFeatures(self, t, df, **kwargs):
        Strategy.genFeatures(self, t, df, **kwargs)
        cols = self.lag_cols + [t]
        ld = self.ld
        # print('??????', self.stationId)
        if ld is True:
            print('London city')
            cols.remove('weather_clu')
        features = []
        for c in cols:
            features += [_ for _ in df.columns if _.find(lag_format(c, '')) != -1]
        features += ['hour'] + cols#, 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed']
        if ld is False:
            features.append('rain_hours')
        features.remove(t)
        return features
class Strategy11(Strategy10):
    def func(self, x):
        return np.sum(x)
    def getW(self, df):
        # print((df.rain_hours * self.DECAY + self.DECAY).min())
        # print((df.rain_hours * self.DECAY + self.DECAY).max())
        # print((df.rain_hours * self.DECAY + self.DECAY).mean())
        # plt.hist((df.rain_hours * self.DECAY + self.DECAY))
        # plt.show()
        return #df.rain_hours * self.DECAY + self.DECAY)
class Strategy12(Strategy10):
    def getW(self, df):
        if self.ld is True or self.target == 'O3':
            return None
        else:
            return df.rain_hours * self.DECAY + self.DECAY
    def genFeatures(self, t, df, **kwargs):
        features = Strategy10.genFeatures(self, t, df, **kwargs)
        if self.ld is True or self.target == 'O3':
            try:
                features.remove('rain_hours')
            except:
                pass
        if self.target == 'O3': # no cluster
            print('no cluster')
            res1 = []
            for f in features:
                if f.startswith('weather_clu'):
                    res1.append('weather'+f[len('weather_clu'):])
                else:
                    res1.append(f)
            features = res1
        return features
    def split(self, df, ratio):
        df_train, df_test = Strategy10.split(self, df, ratio)
        if self.ld is False and self.target != 'O3':
            df_train = df_train[df_train.utc_time >= '2017-04-01']
        return df_train, df_test
class Strategy13(Strategy1):
    def getW(self, df):
        return 1 - 0.99 * df[self.target+'na']
class Strategy14(Strategy1):
    def split(self, df, ratio):
        pivot = pd.to_datetime('2018-05-01')
        st = pd.to_datetime('2017-04-01')
        ed = pd.to_datetime('2017-07-01')
        df_train = df.loc[(df['utc_time'] < ed) & (df['utc_time'] >= st)]
        df_test = df.loc[df['utc_time'] >= pivot]
        if not self.args.deploy:
        #     # df_test = df.loc[df['utc_time'] >= pd.to_datetime('2018-05-01')]
            df_test = df_test.loc[df_test['utc_time'] < pd.to_datetime('2018-05-20')] # use the same dataset as validation set
            # df_test = df_test.loc[df_test['utc_time'] < pd.to_datetime('2018-05-04')] # use the same dataset as validation set
        return df_train, df_test
    def genFeatures(self, *args, **kwargs):
        res = Strategy1.genFeatures(self, *args, **kwargs)
        res1 = []
        for f in res:
            if f.startswith('weather'):
                res1.append('weather_clu'+f[len('weather'):])
            else:
                res1.append(f)
        return res1
class Strategy15(Strategy1):
    def split(self, df, ratio):
        pivot = pd.to_datetime('2018-05-01')
        df_train = df.loc[df['utc_time'].dt.month.isin([9,8,7,6,4])]
        df_train = df_train.loc[df_train['utc_time'] < pivot]
        df_test = df.loc[df['utc_time'] >= pivot]
        if not self.args.deploy:
        #     # df_test = df.loc[df['utc_time'] >= pd.to_datetime('2018-05-01')]
            df_test = df_test.loc[df_test['utc_time'] < pd.to_datetime('2018-05-21')] # use the same dataset as validation set
            # df_test = df_test.loc[df_test['utc_time'] < pd.to_datetime('2018-05-04')] # use the same dataset as validation set
        return df_train, df_test
    def genFeatures(self, *args, **kwargs):
        res = Strategy1.genFeatures(self, *args, **kwargs)
        res1 = []
        for f in res:
            if f.startswith('weather'):
                res1.append('weather_clu'+f[len('weather'):])
            else:
                res1.append(f)
        return res1
class Strategy17(Strategy1):
    def __init__(self, *args, **kwargs):
        Strategy1.__init__(self, *args, **kwargs)
        self.best_params = {}
        self.search_params()
    def search_params(self):
        args = self.args
        for city in ['bj', 'ld']:
            self.setStation(self.dataset[self.dataset.city == city].stationId.values[0])
            for target in ['PM2.5', 'PM10', 'O3']:
                if city == 'ld' and target == 'O3': continue
                param_key = city+'-'+target
                print('searching for '+param_key)
                score_path = os.path.join(args.out, 'score-'+param_key+'.csv')
                if os.path.isfile(score_path):
                    print('Saved file found. Loading')
                    scores = pd.read_csv(score_path).sort_values(by=['mean_test_score'], ascending=False)
                else:
                    print('Saved file not found. Searching')
                    df = self.dataset
                    df = df[df.city == city]
                    features = self.genFeatures(target, df)
                    df = df.dropna(subset=features + [target])
                    print('features=')
                    print(features)
                    if not self.args.deploy:
                        df = df[df.utc_time < '2018-05-21']
                    df = df.sample(n=len(df[df.stationId == self.stationId]))
                    print('len(df)=',len(df))
                    y = df[target]
                    X = df[features]
                    X = X.reset_index(drop=True)
                    y = y.reset_index(drop=True)
                    df = df.reset_index(drop=True)
                    param = Strategy1.getParams(self, df, features, target)[0]

                    # return {
                    #     # 'max_bin': args.max_bin,
                    #     'num_threads': args.num_threads,
                    #     'boosting_type': args.boosting,
                    #     'objective': args.objective,
                    #     'min_data_in_leaf': args.min_data_in_leaf,
                    #     # 'metric': 'na',
                    #     'num_leaves': args.num_leaves,
                    #     'learning_rate': args.lr,
                    #     'feature_fraction': 0.9,
                    #     'bagging_fraction': 0.8,
                    #     'bagging_freq': 5,
                    #     'verbose': 0
                    # }
                    myCViterator = []
                    nFolds = 1
                    pivots = ['2018-05-01']
                    for i in range(nFolds):
                        trainIndices = df[ df['utc_time']<pivots[i] ].index.values.astype(int)
                        testIndices =  df[ df['utc_time']>=pivots[i] ].index.values.astype(int)
                        myCViterator.append( (trainIndices, testIndices) )
                    estimator = lgb.LGBMRegressor(
                        n_jobs=1,
                        boosting_type=param['boosting_type'],
                        objective=param['objective'],
                        min_child_samples=param['min_data_in_leaf'],
                        num_leaves=param['num_leaves'],
                        learning_rate=param['learning_rate'],
                        colsample_bytree=param['feature_fraction'],
                        subsample=param['bagging_fraction'],
                        subsample_freq=param['bagging_freq'],
                        silent=False
                    )
                    param_grid = {
                        'learning_rate': [0.05, 0.25, 0.01],
                        'n_estimators': [args.max_iter],
                        'num_leaves': [31, 63, 127],
                        # 'early_stopping_rounds': [0, args.es],
                        'colsample_bytree' : [0.8, 0.9, 1],
                        'subsample' : [0.8, 0.9, 1],
                        'subsample_freq' : [0, 2, 5],
                        'reg_alpha' : [0,1],
                        'reg_lambda' : [0,1],
                    }
                    smape_loss = make_scorer(_smape, greater_is_better=False)
                    X_val = df[df.utc_time >= pivots[-1]][features]
                    y_val = df[df.utc_time >= pivots[-1]][target]
                    fit_params = {
                        'eval_set': [(X_val, y_val)],
                        'eval_metric': _smape1,
                        'early_stopping_rounds': args.es if args.es > 0 else None,
                        'verbose': False,
                    }
                    gbm = GridSearchCV(estimator, param_grid, scoring=smape_loss, 
                        verbose=10, fit_params=fit_params, cv=myCViterator, 
                        pre_dispatch=args.num_threads, n_jobs=args.num_threads)
                    gbm.fit(X, y)
                    print(gbm.best_params_)
                    print(gbm.best_score_)
                    scores = pd.DataFrame(gbm.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
                    scores.to_csv(score_path, index=False)
                print(scores)
                self.best_params[param_key] = scores.params.values[:10]
                print(self.best_params[param_key])

    def getParams(self, df, features, target, *pargs, **kwargs):
        param = Strategy1.getParams(self, df, features, target)[0]
        param_key = self.city+'-'+target
        ds = self.best_params[param_key]
        res = []
        for d in ds:
            mapper = {
                'colsample_bytree': 'feature_fraction',
                'subsample_freq': 'bagging_freq',
                'subsample': 'bagging_fraction',
            }
            if type(d) is str:
                d = ast.literal_eval(d)
            d = {mapper.get(k, k) : v for k, v in d.items() }
            p = copy.copy(param)
            p.update(d)
            res.append(p)
        return res
class Strategy18(Strategy1):
    def genFeatures(self, t, df, **kwargs):
        features = Strategy1.genFeatures(self, t, df, **kwargs)
        if t == 'O3':
            bak = copy.copy(features)
            for f in bak:
                match = re.match('(.*?)_lag_(\d+)', f)
                if match is not None and int(match.group(2)) > 1:
                    print(f, 'removed')
                    features.remove(f)
        return features  

def getStrategy(name, args):
    if name == 'Strategy1':
        return Strategy1(args)
    elif name == 'Strategy2':
        return Strategy2(args)
    elif name == 'Strategy3':
        return Strategy3(args)
    elif name == 'Strategy4':
        return Strategy4(args)
    elif name == 'Strategy5':
        return Strategy5(args)
    elif name == 'Strategy6':
        return Strategy6(args)
    elif name == 'Strategy7':
        return Strategy7(args)
    elif name == 'Strategy8':
        return Strategy8(args)
    elif name == 'Strategy9':
        return Strategy9(args)
    elif name == 'Strategy10':
        return Strategy10(args)
    elif name == 'Strategy11':
        return Strategy11(args)
    elif name == 'Strategy12':
        return Strategy12(args)
    elif name == 'Strategy13':
        return Strategy13(args)
    elif name == 'Strategy14':
        return Strategy14(args)
    elif name == 'Strategy15':
        return Strategy15(args)
    elif name == 'Strategy17':
        return Strategy17(args)
    elif name == 'Strategy18':
        return Strategy18(args)
    else:
        raise ValueError("No such strategy")