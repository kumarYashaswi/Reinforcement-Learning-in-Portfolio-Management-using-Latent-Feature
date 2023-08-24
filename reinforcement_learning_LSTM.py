# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:34:02 2020

@author: Kumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.nonparametric.smoothers_lowess import lowess

prices_districts =  pd.read_csv('/Users/Kumar/Desktop/PortfolioOptimization/final_data.csv')
prices_districts['date'] =  pd.to_datetime(prices_districts['date'], format='%Y-%m-%d')

dates= prices_districts['date']

start_date_num = prices_districts.iloc[0:1,0]
end_date_num = prices_districts.iloc[-1,0]

prices_districts = prices_districts.set_index(['date'])



prices_districts.loc[datetime.date(year=start_date_num.year, month=start_date_num.month, day=1): 
    datetime.date(year=end_date_num.year, month=end_date_num.month, day=1)].resample('M').mean()
nrows=len(prices) 

def isnotempty(x):
    return not math.isnan(x)

for dn in range(0,70,2):
  for rown in range(0,nrows):
    if (isnotempty(prices.iloc[rown,dn+1]) & (prices.iloc[rown,dn+1] < 4.0) ): 
        prices.iloc[rown,dn] = float('nan')

prices

prices=prices.fillna(prices.rolling(4,min_periods=4).mean())

prices= prices.iloc[:,range(0,70,2)]
pricesmoothed = np.zeros((nrows,len(prices.columns)))

j=0
for dn in range(0,len(prices.columns)):
    y= lowess(prices.iloc[:,dn],range(1,len(prices)+1) , is_sorted=True, frac=0.02)
    pricesmoothed[:,j]=y[:,1]
    j=j+1
    
pricesmoothed=pd.DataFrame(pricesmoothed)
pricesmoothed.columns=prices.columns
pricesmoothed['out']= dates

dist=3
plt.plot(prices.iloc[:,dist:dist+1],color='blue')
plt.plot(pricesmoothed.iloc[:,dist:dist+1],color='red')
plt.title('Orignal Prices')
plt.legend(['OrignalPrice', 'SmoothedPrice'])
plt.xlabel('Months')
plt.ylabel('Price');   

pricesmoothed


