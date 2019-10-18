#!/usr/bin/env python
# coding: utf-8

# In[2]:


#初始库调用
import pandas as pd
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',1000)
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from numpy import concatenate
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt

#下载道琼斯指数数据,只可惜数据有问题
#import yfinance as yf
#data=yf.download("DJI",start="2009-10-16",end="2019-10-16")
#上面代码下载的数据有问题，因此直接在Yahoo Finance上下载csv文件，然后本地调用
data=read_csv('D:\Python Learning\DJI.csv',index_col='Date')
#删除不需要的数据
data.drop(['Adj Close','Volume'],axis=1,inplace=True)
#提取data中的数值，并转化为浮点型
values=data.values
values=values.astype('float32')
# 特征的归一化处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

#将数据变成有监督学习数据，用n_in天之前的数据来预测n_out天之后的数据
def series_to_supervised(data,n_in=1,n_out=1,dropnan=True):
    n_vars=1 if type(data) is list else data.shape[1]
    df=DataFrame(data)
    cols,names=list(),list()
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names+=[('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i==0:
            names+=[('var%d(t)' % (j+1))for j in range(n_vars)]
        else:
            names+=[('var%d(t+%d)' % (j+1,i)) for j in range(n_vars)]
    agg=concat(cols,axis=1)
    agg.columns=names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
#对数据进行有监督学习的转换，用1天的数据来进行预测。由于只预测收盘价，因此删除多余的预测数据
reframed=series_to_supervised(scaled,1,1)
reframed.drop(reframed.columns[[4,5,6]],axis=1,inplace=True)
#分割训练数据和测试数据
values=reframed.values
train=values[:2000,:]
test=values[2000:,:]
#对数据进行形状转化，将dataframe的2维数据转变为lstm输入要求的3维数据，分别为数据总条数，天数，变量的维度
train_x,train_y=train[:,:-1],train[:,-1]
test_x,test_y=test[:,:-1],test[:,-1]
train_x=train_x.reshape((train_x.shape[0],1,train_x.shape[1]))
test_x=test_x.reshape((test_x.shape[0],1,test_x.shape[1]))
#设置神经网络参数，其中隐层维度为64（也是权值矩阵的列数），
model=Sequential()
model.add(LSTM(64,input_shape=(train_x.shape[1],train_x.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(1,activation='relu'))
model.compile(loss='mae',optimizer='adam')
history=model.fit(train_x,train_y,epochs=50,batch_size=100,validation_data=(test_x,test_y),verbose=2,shuffle=False)
#对测试集进行预测，获得预测的收盘价
y_predict=model.predict(test_x)
#对预测值进行反归一化，这里注意反归一化矩阵的尺寸要和归一化一致，因此将预测值与其他数据拼成一个和归一前一样尺寸的矩阵
test_x=test_x.reshape((test_x.shape[0],test_x.shape[2]))
inv_y_test=concatenate((test_x[:,:-1],y_predict),axis=1)
inv_y_test=scaler.inverse_transform(inv_y_test)
inv_y_predict=inv_y_test[:,-1]
#对真实值进行反归一化
test_y = test_y.reshape((len(test_y),1))
inv_y_train = concatenate((test_x[:,:-1],test_y),axis=1)
inv_y_train = scaler.inverse_transform(inv_y_train)
inv_y = inv_y_train[:,-1]
#绘制真实值和预测值的曲线图
plt.plot(inv_y,color='red',label='Original')
plt.plot(inv_y_predict,color='green',label='Predict')
plt.xlabel('the number of test data')
plt.ylabel('Dow Jowns Index')
plt.title('2017.09.27—2019.10.16')
plt.legend()
plt.show()
#回归评价指标
# calculate MSE 均方误差
mse=mean_squared_error(inv_y,inv_y_predict)
# calculate RMSE 均方根误差
rmse = sqrt(mean_squared_error(inv_y, inv_y_predict))
#calculate MAE 平均绝对误差
mae=mean_absolute_error(inv_y,inv_y_predict)
#calculate R_square
r_square=r2_score(inv_y,inv_y_predict)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)
print('r_square: %.6f' % r_square)


# In[ ]:





# In[ ]:




