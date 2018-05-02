import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visAll(aq_info, suf=''):
    aq_info.utc_time = pd.to_datetime(aq_info.utc_time)
    for city in ['bj', 'ld']:
        for c in set(aq_info.columns)-set(['utc_time', 'city', 'stationId', 'meo', 'grid', 'weather']):
            print(city, c)
            plt.clf()
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

if __name__ == '__main__':
    aq_info = pd.read_csv('./data/data_na.csv')
    aq_info_fillna = pd.read_csv('./data/data.csv')
    visAll(aq_info_fillna)
    visAll(aq_info, 'has_na')