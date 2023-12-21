#!/usr/bin/env python
# coding: utf-8

# # 时间序列预测

# In[67]:


import csv, sys, os, math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')

workdir = 'D:\RNAseq\Experiment\PatternRP\work'
data = pd.read_csv(r"C:\Users\14816\Desktop\project\Covid_TimeSeriesAnalysis\clean_data.csv", index_col=[0], parse_dates=[0])


# In[68]:


data


# # 季节性分析

# In[133]:


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

series = data['Confirmed']
result = seasonal_decompose(series, model='additive', period=12)

# 绘制结果
plt.figure(figsize=(30, 15))
result.plot()
plt.show()

组成部分：趋势、季节性、周期性和残差
# # Train/Test Split
# ### Cut off the data after 2021-05-01 to use as our validation set.

# In[69]:


split_date = '2021-05-01'
train = data.loc[data.index < split_date].copy()
test = data.loc[data.index >= split_date].copy()


# In[70]:


train


# In[71]:


test


# In[72]:


def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    #df['hour'] = df['date'].dt.hour
    #df['dayofweek'] = df['date'].dt.dayofweek
    #df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    #df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    #df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['year','month','dayofmonth']]
    if label:
        y = df[label]
        return X, y
    return X


# In[73]:


X_train, y_train = create_features(train, label='Confirmed')
X_test, y_test = create_features(test, label='Confirmed')


# In[74]:


X_test.shape


# # Create Model
# ## XGBoost

# In[57]:


xgbreg = xgb.XGBRegressor(n_estimators = 100, early_stopping_rounds=10)
xgbreg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        #early_stopping_rounds = 50,
       verbose = True)


# In[58]:


plot_importance(xgbreg, height=0.9)

模型最依赖月份和日期进行预测，年份的贡献相对较小。因此说明可以用月日数据预测不同年份的月日数据。【这里是我猜的，可以看看有没有支持文献^-^】
# In[36]:


X_test


# In[59]:


test['Confirmed_Prediction'] = xgbreg.predict(X_test)


# In[38]:


test


# ## Error Metrics On Test Set
# ### Our RMSE error is 3446373669.1810346
# ### Our MAE error is 58090.724137931036
# ### Our MAPE error is 1.543%

# In[60]:


mean_squared_error(y_true=test['Confirmed'],
                   y_pred=test['Confirmed_Prediction'])


# In[61]:


mean_absolute_error(y_true=test['Confirmed'],
                   y_pred=test['Confirmed_Prediction'])


# In[62]:


# MAPE看预测的偏差程度 这里要引用一下mean absolute percent error的文献~
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[63]:


mean_absolute_percentage_error(y_true=test['Confirmed'],
                   y_pred=test['Confirmed_Prediction'])


# ## Worst and Best Predicted Days

# In[64]:


test['error'] = test['Confirmed'] - test['Confirmed_Prediction']
test['abs_error'] = test['error'].apply(np.abs)
error_by_day = test.groupby(['year','month','dayofmonth']) \
    .mean(numeric_only=True)[['Confirmed','Confirmed_Prediction','error','abs_error']]
error_by_day.sort_values('error', ascending=True).head(10)


# ## Prophet

# In[42]:


from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[43]:


# rename
train.reset_index() \
    .rename(columns={'ObservationDate':'ds',
                     'Confirmed':'y'}).head()


# In[44]:


proreg = Prophet()
proreg.fit(train.reset_index() \
              .rename(columns={'ObservationDate':'ds',
                               'Confirmed':'y'}))


# In[46]:


Predict_train = proreg.predict(df=test.reset_index() \
                                   .rename(columns={'ObservationDate':'ds'}))


# In[47]:


Predict_train


# ## Error Metrics
# ### Our RMSE error is 5261817327.887729
# ### Our MAE error is 69814.2196623891
# ### Our MAPE error is 1.85%
# 
# #### comparison in the XGBoost model our errors were significantly high

# In[51]:


mean_squared_error(y_true=test['Confirmed'],
                   y_pred=Predict_train['yhat'])


# In[50]:


mean_absolute_error(y_true=test['Confirmed'],
                   y_pred=Predict_train['yhat'])


# In[52]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=test['Confirmed'],
                   y_pred=Predict_train['yhat'])


# # 传统时间序列预测模型对比

# ## 平稳性检测——单位根检验 ： ADF检验（Augmented Dickey-Fuller Test）

# ### Data Reset

# In[88]:


X_test


# In[79]:


from statsmodels.tsa.stattools import adfuller

result = adfuller(y_train)

print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'\t{key}: {value}')

结论:
由于 ADF 统计量没有低于任何临界值，并且 p-值远高于常用的显著性水平（例如 0.05），因此我们不能拒绝零假设。这表明时间序列数据很可能是非平稳的。
# ## SARIMA

# In[91]:


series = train['Confirmed']


# In[109]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
y_train.index = pd.date_range(start='2020-01-26', periods=423, freq='D')
SARIMAXmodel = SARIMAX(series, order=(1, 2, 1), 
                       seasonal_order=(1, 1, 1, 12))

SARIMAXreg = SARIMAXmodel.fit()
yhat = SARIMAXreg.forecast(steps=67) # 预测未来30天，5月1日到5月31日
yhat = yhat[38:]
print(yhat)


# In[162]:


# 绘制原始数据和预测结果
plt.figure(figsize=(10, 6))
plt.plot(data, label='Actual')
plt.plot(yhat, label='Forecast')
plt.title('SARIMA Method')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.legend()
plt.show()

order=(1, 2, 1)，其中的第二个参数默认是 1 表示对数据进行了一阶差分。如果这不足以使数据平稳，因此用了更高阶的差分。
# ## Error Metrics
# ### Our RMSE error is 6601467840.640008
# ### Our MAE error is 81093.91063713128
# ### Our MAPE error is 2.15%

# In[110]:


mean_squared_error(y_true=test['Confirmed'],
                   y_pred=yhat)


# In[111]:


mean_absolute_error(y_true=test['Confirmed'],
                   y_pred=yhat)


# In[112]:


mean_absolute_percentage_error(y_true=test['Confirmed'],
                   y_pred=yhat)


# ## 霍尔特-温特斯季节性模型

# In[156]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# 选择数据列
hwdata = train['Confirmed']

# 创建并拟合霍尔特-温特斯季节性模型
# 参数 seasonal_periods 根据数据的季节性周期进行调整
hwmodel = ExponentialSmoothing(hwdata, trend='add', seasonal='add', seasonal_periods=12).fit()

# 进行预测
pred = hwmodel.forecast(steps=67)


# In[157]:


# 绘制原始数据和预测结果
plt.figure(figsize=(10, 6))
plt.plot(data, label='Actual')
plt.plot(pred, label='Forecast')
plt.title('Holt-Winters Seasonal Method')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.legend()
plt.show()


# In[158]:


pred = pred[38:]


# ## Error Metrics
# ### Our RMSE error is 5828118112.657714
# ### Our MAE error is 76263.77711225502
# ### Our MAPE error is 2.024%

# In[159]:


mean_squared_error(y_true=test['Confirmed'],
                   y_pred=pred)


# In[160]:


mean_absolute_error(y_true=test['Confirmed'],
                   y_pred=pred)


# In[161]:


mean_absolute_percentage_error(y_true=test['Confirmed'],
                   y_pred=pred)


# In[ ]:




