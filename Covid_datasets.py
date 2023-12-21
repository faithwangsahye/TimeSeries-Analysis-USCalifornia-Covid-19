#!/usr/bin/env python
# coding: utf-8

# In[28]:


import csv, sys, os, math
import pandas as pd
import numpy as np
import plotly.express as px

workdir = 'D:\RNAseq\Experiment\PatternRP\work'


# In[29]:


covid_data = pd.read_csv(r"D:\RNAseq\Experiment\PatternRP\work\covid\archive\covid_19_data.csv")


# In[5]:


covid_data


# In[30]:


cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

#Active Case = Confirmed - Deaths - Recovered
covid_data['Active'] = covid_data['Confirmed'] - covid_data['Deaths'] - covid_data['Recovered']

#Filling missing values
covid_data[['Province/State']] = covid_data[['Province/State']].fillna('')
covid_data[cases] = covid_data[cases].fillna(0)
covid_data.head()


# # WHY TO CHOOSE US FOR ANALYSIS

# In[32]:


#Creating a dataframe with total no of cases for every country
confirmed_cases = pd.DataFrame(covid_data.groupby('Country/Region')['Confirmed'].sum())
confirmed_cases['Country/Region'] = confirmed_cases.index 
confirmed_cases.index = np.arange(1, 230) 

death_cases = pd.DataFrame(covid_data.groupby('Country/Region')['Deaths'].sum())
death_cases['Country/Region'] = death_cases.index
death_cases.index = np.arange(1, 230)

active_cases = pd.DataFrame(covid_data.groupby('Country/Region')['Active'].sum())
active_cases['Country/Region'] = active_cases.index
active_cases.index = np.arange(1, 230)

global_confirmed_cases = confirmed_cases[['Country/Region', 'Confirmed']]
global_death_cases = death_cases[['Country/Region', 'Deaths']]
global_active_cases = active_cases[['Country/Region', 'Active']]

fig = px.bar(global_confirmed_cases.sort_values('Confirmed', ascending=False)[:20][::-1], x='Confirmed', y='Country/Region', title='Confirmed Cases Worldwide', text='Confirmed', height=900, orientation='h')
fig.show()
fig = px.bar(global_death_cases.sort_values('Deaths', ascending=False)[:20][::-1], x='Deaths', y='Country/Region', title='Death Cases Worldwide', text='Deaths', height=900, orientation='h')
fig.show()
fig = px.bar(global_active_cases.sort_values('Active', ascending=False)[:20][::-1], x='Active', y='Country/Region', title='Active Cases Worldwide', text='Active', height=900, orientation='h')
fig.show()


# ## 筛选出美国数据

# In[33]:


us_data = covid_data[covid_data['Country/Region'] == 'US'].copy()
us_data['ObservationDate'] = pd.to_datetime(us_data['ObservationDate']) # 计算每天的累积确诊、死亡和恢复病例数
us_data = us_data.reset_index()


# In[22]:


us_data.reset_index()
#us_data.to_csv('US_covid_data.csv')


# # WHY TO CHOOSE California FOR ANALYSIS

# In[36]:


confirmed_cases = pd.DataFrame(us_data.groupby('Province/State')['Confirmed'].sum())

confirmed_cases['Province/State'] = confirmed_cases.index 
confirmed_cases.index = np.arange(1, 200) 

death_cases = pd.DataFrame(us_data.groupby('Province/State')['Deaths'].sum())
death_cases['Province/State'] = death_cases.index
death_cases.index = np.arange(1, 200)

active_cases = pd.DataFrame(us_data.groupby('Province/State')['Active'].sum())
active_cases['Province/State'] = active_cases.index
active_cases.index = np.arange(1, 200)

global_confirmed_cases = confirmed_cases[['Province/State', 'Confirmed']]
global_death_cases = death_cases[['Province/State', 'Deaths']]
global_active_cases = active_cases[['Province/State', 'Active']]

fig = px.bar(global_confirmed_cases.sort_values('Confirmed', ascending=False)[:20][::-1], x='Confirmed', y='Province/State', title='Confirmed Cases US', text='Confirmed', height=900, orientation='h')
fig.show()
fig = px.bar(global_death_cases.sort_values('Deaths', ascending=False)[:20][::-1], x='Deaths', y='Province/State', title='Death Cases US', text='Deaths', height=900, orientation='h')
fig.show()
fig = px.bar(global_active_cases.sort_values('Active', ascending=False)[:20][::-1], x='Active', y='Province/State', title='Active Cases US', text='Active', height=900, orientation='h')
fig.show()


# In[38]:


cf_data = us_data[us_data['Province/State'] == 'California'].copy()
cf_data['ObservationDate'] = pd.to_datetime(cf_data['ObservationDate'])
cf_data = cf_data.reset_index(drop=True)
#cf_data.to_csv('cf_covid.csv')


# In[42]:


cf_data.reset_index(drop=True)


# ## Final Data For Modeling

# In[50]:


cf_cumulative = cf_data.groupby('ObservationDate').sum(numeric_only=True) # 计算每天的累积确诊、死亡和恢复病例数
cf_cumulative = cf_cumulative[['Confirmed', 'Deaths', 'Recovered']] # 只保留需要的列


# In[51]:


cf_cumulative


# In[52]:


cf_cumulative.to_csv('clean_data.csv')

