import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as st
import numpy as np
import pyflux as pf
import matplotlib.pyplot as plt

def visAll(aq_info, suf=''):
    aq_info.utc_time = pd.to_datetime(aq_info.utc_time)
    for city in ['bj', 'ld']:
        for c in set(aq_info.columns)-set(['utc_time', 'city', 'stationId', 'meo', 'grid', 'weather']):
            print(city, c)
            plt.figure()
            try:
                df=pd.pivot_table(aq_info[aq_info['city']==city].reset_index(),
                               index='utc_time', columns='stationId', values=c
                               )
#                 df.index=pd.to_datetime(df.index)
                ax=df.plot(subplots=True, figsize=(128,64), title='./fig/{}_{}'.format(city, c))
#                 # set monthly locator
# #                 ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#                 # set formatter
#                 ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#                 # set font and rotation for date tick labels
#                 plt.gcf().autofmt_xdate()

            except Exception as e:
                print(e)

            plt.savefig('./fig/{}_{}_{}.png'.format(city, c, suf))