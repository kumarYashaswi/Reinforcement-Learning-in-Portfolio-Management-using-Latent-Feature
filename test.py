# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:10:29 2021

@author: KUMAR YASHASWI
"""

self.codes=codes
codes = ["AAPL", "AXA", "BRH", "CITI", "GILEAD","HON", "INTU", "JPM", "NKE", "NVDA","ORCL", "PG", "UAL", "WMT", "XOM"]
market = 'America1'
features=["close","high","low"]
        data = pd.read_csv(r'./data/' + market + '.csv', index_col=0, parse_dates=True, dtype=object)
        data["code"] = data["code"].astype(str)
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        print('wwwwwwwww')

    data.loc[0:1, 'time']
    (start_time,end_time)= (datetime.datetime(2015, 1, 1),datetime.datetime(2018, 12, 31))    
    (start_time,end_time)= (datetime.datetime(2007, 1, 1),datetime.datetime(2015, 12, 31)) 
    (start_time_Zoom,end_time_Zoom)= (datetime.datetime(2007, 1, 1),datetime.datetime(2019, 12, 31))
        data[features]=data[features].astype(float)
        data_zoom=data[start_time_Zoom.strftime("%Y-%m-%d"):end_time_Zoom.strftime("%Y-%m-%d")]
        data=data[start_time.strftime("%Y-%m-%d"):end_time.strftime("%Y-%m-%d")]
        data=data
        #TO DO:REFINE YOUR DATA


SVD_list(data_zoom.reset_index(),60)
data_zoom=data_zoom.reset_index()
data_zoom['close'].iloc[0:1,:].values





        #Initialize parameters
        M=15+1
        N=3
        L=int(60)
        L1 = 15
        date_set=pd.date_range(start_time,end_time)
        encoder_conv = autoencodetrain()
        #为每一个资产生成数据
        asset='AAPL'
        asset_dict=dict()#每一个资产的数据
        for asset in codes:
            asset_data=data[data["code"]==asset].reindex(date_set).sort_index()#加入时间的并集，会产生缺失值pd.to_datetime(self.date_list)
            asset_data=asset_data.resample('D').mean()
            asset_data['close']=asset_data['close'].fillna(method='pad')
            
            #base_price = asset_data['close'].loc[asset_data.shape[0]-1, 'close']
            base_price = asset_data['close'].iloc[-1]
            asset_dict[str(asset)]= asset_data
            asset_dict[str(asset)]['close'] = asset_dict[str(asset)]['close'] / base_price

            if 'high' in features:
                asset_dict[str(asset)]['high'] = asset_dict[str(asset)]['high'] / base_price

            if 'low' in features:
                asset_dict[str(asset)]['low']=asset_dict[str(asset)]['low']/base_price

            if 'open' in features:
                asset_dict[str(asset)]['open']=asset_dict[str(asset)]['open']/base_price

            asset_data=asset_data.fillna(method='bfill',axis=1)
            asset_data=asset_data.fillna(method='ffill',axis=1)#根据收盘价填充其他值
            asset_data=asset_data.fillna(method='bfill',axis=0)
            #***********************open as preclose*******************#
            #asset_data=asset_data.dropna(axis=0,how='any')
            asset_dict[str(asset)]=asset_data

        
SVD_dict=dict()
for feat in features:
    SVDData=getSVDData(asset_dict,feat,codes)
    SVDData=SVDData*100
    U_list, S_list ,V_list=SVD_list(SVDData,L-1)
    SVD_dict[str(feat)]=[U_list, S_list ,V_list]
  
        
Cov_dict=dict()
for feat in features:
    CovData=getSVDData(asset_dict,feat,codes)
    CovData=CovData*100
    Cov_dict[str(feat)]=CovData
    

ZoomSVD(SVD_dict,t - L - 1,t - 2,'close',L)

t1=5
t2=64
feature='close'
i=2
data= Cov_dict[str(feature)]
data=data.iloc[t1:t2]
    data1=np.array(data.cov())
    s=pd.Series(data1[i,:]) 
ZoomSVD(SVD_dict,t1,t2,feature,b):
    [U_list,S_list,V_list]= SVD_dict[str(feature)]
    block_diagnol,  S, V=partial_svd(U_list,S_list,V_list,t1,t2, L-1)
    svd_encode=np.dot(np.diag(S), V.T)
    s=pd.Series(svd_encode[:,1])   
    return s
 
s=pd.Series(svd_encode[:,1])   
 
s=np.diag(S)
    
L=60
U_list, S_list ,V_list=SVD_list(SVDData,L)


asset_data=asset_dict[str('AAPL')]
asset_data=asset_data.reset_index()
 b= asset_data.loc[t - L - 1:t - 2, 'close']         
L
s=asset_data.loc[t - L - 1:t - 2, 'close']
if(self.L1==self.L):
            states=[]
            price_history=[]
            t =L+1
            length=len(date_set)
            while t<length-1:
                V_close = np.ones(L)
                if 'high' in features:
                    V_high=np.ones(L)
                if 'open' in features:
                    V_open=np.ones(L)
                if 'low' in features:
                    V_low=np.ones(L)
    
    
                y=np.ones(1)
                state=[]
                for asset in codes:
                    asset_data=asset_dict[str(asset)]
                    asset_data=asset_data.reset_index()
                    V_close = np.vstack((V_close, asset_data.loc[t - L - 1:t - 2, 'close']))
                    if 'high' in features:
                        V_high=np.vstack((V_high,asset_data.loc[t-L-1:t-2,'high']))
                    if 'low' in features:
                        V_low=np.vstack((V_low,asset_data.loc[t-L-1:t-2,'low']))
                    if 'open' in features:
                        V_open=np.vstack((V_open,asset_data.loc[t-L-1:t-2,'open']))
                    y=np.vstack((y,asset_data.loc[t,'close']/asset_data.loc[t-1,'close']))
                state.append(V_close)
                if 'high' in features:
                    state.append(V_high)
                if 'low' in features:
                    state.append(V_low)
                if 'open' in features:
                    state = np.stack((state,V_open), axis=2)
    
                state=np.stack(state,axis=1)
                state = state.reshape(1, M, L, N)
                states.append(state)
                price_history.append(y)
                t=t+1
                
            if(self.L1!=self.L):
                states=[]
                price_history=[]
                t =L+1
                length=len(date_set)
                while t<length-1:
                    V_close = np.ones(L1)
                    if 'high' in features:
                        V_high=np.ones(L1)
                    if 'open' in features:
                        V_open=np.ones(L1)
                    if 'low' in features:
                        V_low=np.ones(L1)
        
        
                    y=np.ones(1)
                    state=[]
                    asset='AAPL'
                    for asset in codes:
                        asset_data=asset_dict[str(asset)]
                        asset_data=asset_data.reset_index()
                        V_close = np.vstack((V_close, autoencode(asset_data.loc[t - L - 1:t - 2, 'close'],encoder_conv)))
                        if 'high' in features:
                            V_high=np.vstack((V_high,autoencode(asset_data.loc[t-L-1:t-2,'high'],encoder_conv)))
                        if 'low' in features:
                            V_low=np.vstack((V_low,autoencode(asset_data.loc[t-L-1:t-2,'low'],encoder_conv)))
                        if 'open' in features:
                            V_open=np.vstack((V_open,autoencode(asset_data.loc[t-L-1:t-2,'open'],encoder_conv)))
                        y=np.vstack((y,asset_data.loc[t,'close']/asset_data.loc[t-1,'close']))
                    state.append(V_close)
                    if 'high' in features:
                        state.append(V_high)
                    if 'low' in features:
                        state.append(V_low)
                    if 'open' in features:
                        state = np.stack((state,V_open), axis=2)
        
                    state=np.stack(state,axis=1)
                    state = state.reshape(1, M, L1, sN)
                    states.append(state)
                    price_history.append(y)
                    t=t+1
                    
                    

if(self.L1!=self.L):
                states=[]
                price_history=[]
                t =L+1
                length=len(date_set)
                while t<length-1:
                    V_close = np.ones(L1)
                    if 'high' in features:
                        V_high=np.ones(L1)
                    if 'open' in features:
                        V_open=np.ones(L1)
                    if 'low' in features:
                        V_low=np.ones(L1)
        
        
                    y=np.ones(1)
                    state=[]
                    asset='AAPL'
                    for asset in codes:
                        asset_data=asset_dict[str(asset)]
                        asset_data=asset_data.reset_index()
                        V_close = np.vstack((V_close, ZoomSVD(asset_data.loc[t - L - 1:t - 2, 'close'],encoder_conv)))
                        if 'high' in features:
                            V_high=np.vstack((V_high,autoencode(asset_data.loc[t-L-1:t-2,'high'],encoder_conv)))
                        if 'low' in features:
                            V_low=np.vstack((V_low,autoencode(asset_data.loc[t-L-1:t-2,'low'],encoder_conv)))
                        if 'open' in features:
                            V_open=np.vstack((V_open,autoencode(asset_data.loc[t-L-1:t-2,'open'],encoder_conv)))
                        y=np.vstack((y,asset_data.loc[t,'close']/asset_data.loc[t-1,'close']))
                    state.append(V_close)
                    if 'high' in features:
                        state.append(V_high)
                    if 'low' in features:
                        state.append(V_low)
                    if 'open' in features:
                        state = np.stack((state,V_open), axis=2)
        
                    state=np.stack(state,axis=1)
                    state = state.reshape(1, M, L1, sN)
                    states.append(state)
                    price_history.append(y)
                    t=t+1