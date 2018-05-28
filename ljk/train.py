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
from strategy import smape, _smape
from numba import jit
try:
    import cPickle as pickle
except:
    import pickle

def check_no_dup(data):
    for s in set(data.stationId):
        ind = data[data['stationId']==s].utc_time
        assert len(ind) == len(set(ind)), (s, Counter(ind))

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
    plt.figure(figsize=(20, 5))
    plt.plot(list(y_pred))
    plt.plot(list(y_real))
    plt.legend(['pred', 'real'])
    plt.title(os.path.join(path, s+'.png'))
    my_x_ticks = np.arange(0, len(y_pred), 48)
    plt.xticks(my_x_ticks)  
    plt.savefig(os.path.join(path, s+'.png'))

# @jit
# @profile
def evaluate(gbm, df_slice, features, t, date_range, genNext):
    print('evaluate date_range={}'.format(date_range))
    print('evaluate features={}'.format(features))
    df_slice_dropna = df_slice.dropna(subset=features)
    st = date_range.min()
    ed = date_range.max()
    if st.hour != 0: st = st + timedelta(days=1) - timedelta(hours=st.hour)
    # ed += timedelta(hours=1)
    # ed -= timedelta(hours=ed.hour)
    date_range = pd.date_range(st, ed + timedelta(days=1), freq='1H')
    print(st, ed)
    y_pred = []
    y_test = []
    scores = []
    tot = 0
    df_test = pd.merge(df_slice, pd.DataFrame({'utc_time': date_range}), how='right', on='utc_time').sort_values(by=['utc_time'])
    
    while st + timedelta(days=1) <= ed:
        y1 = stra.gbm_predict(gbm, df_slice_dropna, 48, st, genNext, features)
        y_pred += y1
        # df_test_slice = df_test[df_test.utc_time >= st].iloc[:48]
        df_test_slice = df_test.iloc[tot / 2:tot/2+48]
        # print('df_test_slice=\n{}'.format(df_test_slice))
        assert df_test_slice.hour.iloc[0] == 0, df_test_slice
        y2 = list(df_test_slice[t+'_raw'])
        y_test += y2
        scores.append([_smape(y1, y2), st])
        assert df_test_slice.utc_time.min() >= st and df_test_slice.utc_time.max() < st + timedelta(hours=48) and \
            df_test_slice.utc_time.is_unique, (st, df_test_slice.utc_time)
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
    sys.stderr = sys.stdout
    mod, score, date_range, importance = stra.train(df_slice, features, t, (args.num_eval, 'num'))
    if not deploy:
        Min = date_range.min()
        date_range = pd.date_range(Min, date_range.max(), freq='1H')
        if args.no_evaluate:
            date_range = pd.date_range(Min, Min, freq='1H')
            print(date_range)
        score, y_pred, y_test, scores = evaluate(mod, df_slice, features, t, date_range, stra.createGenNext(t))
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if not deploy:
        print_prof_data()
        importance['gas'] = t
        importance['stationId'] = s
        importance['city'] = df_slice.iloc[0].city
        import_table = pd.concat([import_table, importance]) if import_table is not None else importance

        score_table[t].append(score)
        scores['city'] = df_slice.iloc[0].city
        scores_table = scores_table.append(scores)
        plot(y_pred, y_test, os.path.join(OutPath, s), t)
        # save_mod(mod, os.path.join(OutPath, s), t)
        print('scores={}'.format(scores))
        print('score={}'.format(score))
        print('mean={}'.format(np.array(filter(lambda x: x==x, score_table[t])).mean()))
    if deploy:
        set_ans(mod, ans)
    return import_table

def getScores2(scores_table):
    scores2 = pd.DataFrame()
    dts = []
    for dt in set(scores_table.columns) - set(['stationId', 'gas', 'city']):
        if scores_table[dt].isna().any(): continue
        dts.append(dt)
    dts = sorted(dts)
    print(dts)
    for dt in dts:
        col = []
        for gas in ['PM2.5', 'PM10', 'O3']:
            bj = scores_table[(scores_table.gas == gas) & (scores_table.city == 'bj')][dt].mean()
            ld = scores_table[(scores_table.gas == gas) & (scores_table.city == 'ld')][dt].mean()
            col.extend([bj, ld])
        bj_mean = (col[0] + col[2] + col[4]) / 3
        ld_mean = (col[1] + col[3]) / 2
        mean = (bj_mean + ld_mean) / 2
        col += [bj_mean, ld_mean, mean]
        scores2[dt] = col
    types = ['{}-{}'.format(city, gas) for gas in ['PM2.5', 'PM10', 'O3'] for city in ['bj', 'ld']] + ['bj-mean', 'ld-mean', 'mean']
    print(types)
    scores2['type'] = types
    return scores2

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
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--thres", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--es", type=int, default=40)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--predict-date", type=str, default=None)
    parser.add_argument("--data-path", type=str, default='../data/data.csv')
    parser.add_argument("--objective", type=str, default='regression')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--boosting", default='gbdt')
    args = parser.parse_args()
    if args.boosting not in ['gbdt', 'rf', 'dart', 'goss']:
        parser.error(args.boosting)
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
        scores2 = getScores2(scores_table)
        scores2.set_index('type').to_csv(os.path.join(OutPath,'./scores2.csv'))
                
    if deploy:
        ans.to_csv(args.outcsv, index=False)
    print('elapsed time={}s'.format(time.time() - st))
if __name__ == '__main__':
    main()
    # print_prof_data()