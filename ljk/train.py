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
import strategy
try:
    import cPickle as pickle
except:
    import pickle

def check_no_dup(data):
    for s in set(data.stationId):
        ind = data[data['stationId']==s].utc_time
        assert len(ind) == len(set(ind)), (s, Counter(ind))

# @profile
def _smape(actual, predicted):
    actual = map(lambda x: np.nan if x < 0 else x, actual)
    predicted = map(lambda x: np.nan if x < 0 else x, predicted)
    a = np.abs(np.array(actual) - np.array(predicted))
#     a[~(a==a)]=0
    b = np.array(actual) + np.array(predicted)
    return 2 * np.nanmean(np.divide(a, b, out=np.zeros_like(a), where=(b!=0), casting='unsafe'))

# @profile
def smape(predicted, train_data):
    actual = train_data.get_label()
    return 'smape', _smape(actual, predicted), False

# def arma_predict(gbm, X_test1, length, genNext):
#     # assert(len(X_test) >= length)
#     y_pred = []
#     # lags = [_ for _ in gbm.feature_name() if _.find(lag_format(target, '')) != -1]
#     # print(lags)
#     for i in range(length):
#         y_pred.append(gbm.predict(X_test1)[0])
#         print(X_test1, y_pred)
#         X_test1 = genNext(X_test1, y_pred[-1])
#         # for j in range(1, len(lags) + 1):
#         #     if i + j >= length: break
#         #     X_test.iloc[i + j][lag_format(target, j)] = y_pred[-1]
#     return y_pred

# @profile
# def split(df, ratio):
#     st = df.utc_time.max()
#     st = st - datetime.timedelta(hours=st.hour)
#     if ratio[1] == 'num': num = ratio[0]
#     else: num = int(n * ratio[0])
#     print(num)
#     while True:
#         hours = pd.date_range(st, st + datetime.timedelta(hours=num-1), freq='1H')
#         if len(df[df.utc_time.isin(hours)]) >= num * 0.8 and len(df[df.utc_time == st]) > 0:
#             tmp = df[df.utc_time.isin(hours)]
#             ind = tmp.index[0]
#             return df.loc[:ind].iloc[:-1], tmp, hours
#         st -= pd.DateOffset(n=1)
#     return df.iloc[:-num], df.iloc[-num:], hours

def split(df, ratio):
    df_train = df.loc[df['utc_time'] <= pd.to_datetime('2018-04-20')]
    # df_train = df.loc[(df['utc_time'] <= pd.to_datetime('2017-06-01')) & (df['utc_time'] >= pd.to_datetime('2017-04-01'))]
    df_test = df.loc[df['utc_time'] > pd.to_datetime('2018-04-20')]
    if not args.deploy:
        # df_test = df.loc[df['utc_time'] >= pd.to_datetime('2018-05-01')]
        # df_test = df_test.loc[df_test['utc_time'] <= pd.to_datetime('2018-05-16')] # use the same dataset as validation set
        df_test = df_test.loc[df_test['utc_time'] < pd.to_datetime('2018-05-04')] # use the same dataset as validation set

    return df_train, df_test

# @profile
def train(df, features, target, ratio):
    assert(ratio[1] == 'num')
    print('target=', target)
    df = df.dropna(subset=features + [target]) # TODO: dropna consecutive violated
    df_train, df_test = split(df, ratio)
    # df_train, df_test = df.iloc[:-ratio[0]], df.iloc[-ratio[0]:]
    assert df_train.utc_time.max() < df_test.utc_time.min(), (df_train.utc_time.max(), df_test.utc_time.min())
    date_range = df_test.utc_time

    print(df_train.tail())
    print('val date_range={}'.format(date_range))

    # if ratio[1] == 'num': n1 = n - ratio[0]
    # else: n1 = int(n * ratio[0])
    # n2 = n - n1
    # df_train = df.iloc[:n1]
    # df_test = df.iloc[n1:] if ratio[0] != 0 else df.iloc[-1:]

    y_train = df_train[target]
    y_test = df_test[target]
    X_train = df_train[features]
    X_test = df_test[features]

    num_train, num_feature = X_train.shape
    print('num_train={}, num_feature={} num_test={}'.format(num_train, num_feature, len(X_test)))

    # create dataset for lightgbm
    # if you want to re-use data, remember to set free_raw_data=False
    lgb_train = lgb.Dataset(X_train, y_train,
                            weight=None, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                        weight=None, free_raw_data=False)

    # specify your configurations as a dict
    params = {
        # 'max_bin': args.max_bin,
        'num_threads': args.num_threads,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        # 'metric': 'na',
        'num_leaves': args.num_leaves,
        'learning_rate': args.lr,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # generate a feature name
    feature_name = df_train.drop(target, axis=1).columns#['feature_' + str(col) for col in range(num_feature)]

    print('Start training...')

    # continue training
    # init_model accepts:
    # 1. model file name
    # 2. Booster()
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=args.max_iter,
                    valid_sets=[lgb_eval, lgb_train],
                    feval=smape,
                    valid_names=['eval', 'train'],
                    early_stopping_rounds=40 if args.es > 0 else None)

    print('Finish 10 - 20 rounds with model file...')

    # feature importances
    importance = list(100 * gbm.feature_importance().astype(np.float) / gbm.feature_importance().sum())
    print('Feature importances:')
    importance = pd.DataFrame({"feature": gbm.feature_name(), "importance": importance})
    print(importance.sort_values(by=['importance']))

    # can predict with any iteration when loaded in pickle way
    # y_pred = gbm.predict(X_test)
    # eval with loaded model
    return gbm, np.nan, date_range, importance
    # if ratio[0] > 0:
    #     y_pred = arma_predict(gbm, X_test.iloc[0], ratio[0], stra.createGenNext(target))
    #     df_test = pd.merge(df_test, pd.DataFrame({'utc_time': date_range}), how='right', on='utc_time')
    #     y_test = df_test[target]
    #     print(y_test)
    #     print(y_pred)
    #     score = _smape(y_test, y_pred)
    #     # print('The rmse of pickled model\'s prediction is:', mean_squared_error(y_test, y_pred) ** 0.5 / y_test.mean())
    #     print('The smape of pickled model\'s prediction is:', score)

    #     return gbm, score, y_pred, y_test, importance
    # else:
    #     return gbm, np.nan, None, None, importance

# @profile
def fix(target_stations, ref):
    res = {}
    for i in range(len(target_stations)):
        res[target_stations[i]] = target_stations[i]
        if target_stations[i].find('_') == -1: continue
        t = target_stations[i].split('_')[0]
        if t == 'miyun': 
            res[target_stations[i]] = 'miyun_aq'
        elif t == 'miyunshuik':
            res[target_stations[i]] = 'miyunshuiku_aq'
        else:
            for s in ref:
                if s[:len(t)] == t:
                    res[target_stations[i]] = s
                    break
    return res
# @profile
def save_mod(gbm, path, s):
    # save model to file
    gbm.save_model(os.path.join(path, 'model_{}.txt'.format(s)))

    # dump model to json (and save to file)
    print('Dump model to JSON...')
    model_json = gbm.dump_model()

    with open(os.path.join(path, 'model_{}.json'.format(s)), 'w+') as f:
        json.dump(model_json, f, indent=4)
# @profile
def plot(y_pred, y_real, path, s):
    plt.clf()
    plt.plot(list(y_pred))
    plt.plot(list(y_real))
    plt.legend(['pred', 'real'])
    plt.title(os.path.join(path, s+'.png'))
    plt.savefig(os.path.join(path, s+'.png'))

# @profile
def evaluate(gbm, df_slice, features, t, date_range, genNext):
    print('evaluate date_range={}'.format(date_range))
    df_slice_dropna = df_slice.dropna(subset=features)
    st = date_range.min()
    ed = date_range.max()
    if st.hour != 0: st = st + timedelta(days=1) - timedelta(hours=st.hour)
    ed += timedelta(hours=1)
    ed -= timedelta(hours=ed.hour)
    date_range = pd.date_range(st, ed, freq='1H')
    print(st, ed)
    y_pred = []
    y_test = []
    scores = []
    tot = 0
    df_test = pd.merge(df_slice, pd.DataFrame({'utc_time': date_range}), how='right', on='utc_time').sort_values(by=['utc_time'])
    while st + timedelta(days=1) < ed:
        y1 = stra.gbm_predict(gbm, df_slice_dropna, 48, st, genNext, features)
        y_pred += y1
        df_test_slice = df_test[df_test.utc_time >= st].iloc[:48]
        # print('df_test_slice=\n{}'.format(df_test_slice))
        y2 = list(df_test_slice[t+'_raw'])
        y_test += y2
        scores.append([_smape(y1, y2), st])
        assert df_test_slice.utc_time.min() >= st and df_test_slice.utc_time.max() < st + timedelta(hours=48) and \
            df_test_slice.utc_time.is_unique, df_test_slice.utc_time
        tot += 48
        st += pd.DateOffset(n=1)
    print('y_test=\n{}'.format(y_test))
    print('y_pred=\n{}'.format(y_pred))
    assert len(y_pred) == tot, ('length not equal', len(y_pred), tot)
    score = _smape(y_test, y_pred)
    print('The smape of pickled model\'s prediction is:', score)
    scores = pd.DataFrame(scores).rename(columns={0: 'score', 1: 'utc_time'}, index=str)
    scores = scores.pivot_table(columns='utc_time', values='score')
    scores['stationId'] = df_slice.stationId.values[0]
    scores['gas'] = t
    scores.reset_index(drop=True, inplace=True)
    print('All score is:', scores)
    return score, y_pred, y_test, scores

# @profile
def work(df, import_table, t, s, _s, score_table, ans):
    global scores_table
    print(t)
    stra.setStation(s)
    df_slice = df.loc[df['stationId']==s]
    if df_slice.city.iloc[0] == 'ld' and t == 'O3':
        score_table[t].append(np.nan)
        return import_table
    # features = genFeatures1(t, df)
    features = stra.genFeatures(t, df)
    def set_ans(gbm, ans):
        # get next midnight's time
        st = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0) #df_slice.dropna(subset=[t]).utc_time.max()
        st = st - timedelta(hours=st.hour) + timedelta(days=1)
        if args.predict_date is not None:
            st = args.predict_date
        print('st={}'.format(st))
        predict = pd.DataFrame(stra.gbm_predict(gbm, df_slice.dropna(subset=features), 48, st, stra.createGenNext(t, deploy=True), features))
        assert len(predict) == 48, 'length != 48'
        # cur_date = df_slice.dropna(subset=[t]).utc_time.max() + pd.DateOffset(hours=1)
        # cur_h = cur_date.hour
        # print('cur_h=',cur_h,cur_date)
        # length = 48 + 24 - cur_h
        # num_lags = len([_ for _ in gbm.feature_name() if _.find(lag_format(t, '')) != -1])
        # hours = pd.DataFrame({'utc_time':pd.date_range(cur_date, cur_date+pd.DateOffset(hours=length), freq='1H')})
        # X_test = pd.merge(hours, df_slice, how='left', on='utc_time')
        # X_test.hour = X_test.utc_time.dt.hour
        # X_test = X_test[features]
        # print(X_test)
        # predict = pd.DataFrame(arma_predict(mod, X_test, length, t))
        print('isna=',predict.isna().any().any())
        print(predict)
        predict = predict.fillna(predict.mean())
        predict = predict.fillna(0)
        print(len(predict))
        print(predict.mean())
        predict = predict[0]
        for i in range(48):
            ans.loc[ans['test_id'] == "%s#%s"%(_s,i), t] = predict[i]
    sys.stdout = open(os.path.join(OutPath, s, '{}.log'.format(t)), 'w')
    mod, score, date_range, importance = train(df_slice, features, t, (args.num_eval, 'num'))
    if not deploy:
        Min = date_range.min()
        date_range = pd.date_range(Min, date_range.max(), freq='1H')
        if args.no_evaluate:
            date_range = pd.date_range(Min, Min, freq='1H')
            print(date_range)
        score, y_pred, y_test, scores = evaluate(mod, df_slice, features, t, date_range, stra.createGenNext(t))
    sys.stdout = sys.__stdout__
    if not deploy:
        print_prof_data()
        importance['gas'] = t
        importance['stationId'] = s
        importance['city'] = df_slice.iloc[0].city
        import_table = pd.concat([import_table, importance]) if import_table is not None else importance

        score_table[t].append(score)
        scores_table = scores_table.append(scores)
        plot(y_pred, y_test, os.path.join(OutPath, s), t)
        # save_mod(mod, os.path.join(OutPath, s), t)
        print('scores={}'.format(scores))
        print('score={}'.format(score))
        print('mean={}'.format(np.array(filter(lambda x: x==x, score_table[t])).mean()))
    if deploy:
        set_ans(mod, ans)
    return import_table

# @profile
def main():
    global OutPath, deploy, args, stra, scores_table
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--outcsv")
    parser.add_argument("--num-eval", type=int, default=24*4)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--strategy", default='Strategy2')
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--thres", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--es", type=int, default=40)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--predict-date", type=str, default=None)
    args = parser.parse_args()

    if args.deploy and args.outcsv is None:
        parser.error('--outcsv is required when --deploy is selected')
    if args.predict_date is not None:
        args.predict_date = datetime.datetime.strptime(args.predict_date,'%Y-%m-%d')
        # from dateutil import tz
        # args.predict_date = args.predict_date.replace(tzinfo=tz.tzutc())
    OutPath = args.out#sys.argv[1]
    deploy = args.deploy

    if not os.path.exists(OutPath):
        os.makedirs(OutPath)

    st = time.time()
    stra = strategy.getStrategy(args.strategy, args)
    df = stra.dataset #create_dataset1('../data/data.csv', MaxLagging=args.lag)

    stations = set(df['stationId'])
    ans = pd.read_csv('../data/submit/sample_submissioin.csv')
    target_stations = list(set(map(lambda x: x.split('#')[0], ans[ans.columns[0]])))
    # ans = pd.read_csv('../../kdd/ans3.csv')
    # ans2 = ans.copy()
    # ans3 = ans.copy()
    fix_map = fix(target_stations, stations)
    print(target_stations, len(target_stations))
    print(fix_map)

    score_table = {'stationId': [], 'PM2.5': [], "PM10":[], "O3": []}
    scores_table = pd.DataFrame()
    import_table = None
    for (idx, _s) in enumerate(target_stations):
        s = fix_map[_s]
        score_table['stationId'].append(s)
        print('=====>%d/%d %s->%s'%(idx, len(target_stations), _s, s))
        if not os.path.exists(os.path.join(OutPath, s)):
            os.mkdir(os.path.join(OutPath, s))
        for t in ['PM2.5', 'PM10', 'O3']:
            import_table = work(df, import_table, t, s, _s, score_table, ans)

    if not deploy:
        pd.DataFrame(score_table).to_csv(os.path.join(OutPath, 'score.csv'), index=False)
        scores_table.to_csv(os.path.join(OutPath, 'scores.csv'), index=False)
        import_table.to_csv(os.path.join(OutPath, 'import_table.csv'), index=False)
        pd.DataFrame(import_table.groupby(['gas', 'feature']).mean()).reset_index()\
            .sort_values(by=['gas', 'importance'], ascending=False).to_csv(os.path.join(OutPath, 'import_table_sorted.csv'), index=False)
    if deploy:
        ans.to_csv(args.outcsv, index=False)
    print('elapsed time={}s'.format(time.time() - st))
if __name__ == '__main__':
    main()
    # print_prof_data()