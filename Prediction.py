#!/usr/bin/env python
# coding: utf-8

# In[16]:


import math 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

today = datetime.now().strftime("%d%m%Y")
filepath = f"/Users/sherbhanu/Desktop/Desiredata {today}.csv"
column_list = ["repurchase_price_per_unit","net_asset_value","outstanding_number_of_units","nav_per_unit","sale_price_per_unit","entry_id","scheme_id","scheme_name"]
list_of_date_columns = ['date_valued']
indexing_column = ['date_valued']

def load_data(filepath, list_of_date_columns, indexing_column,column_types={},remove_duplicates=False):
    df = pd.read_csv(filepath, parse_dates=list_of_date_columns,index_col=indexing_column)
    if column_types:
        for key in column_types.keys():
            if column_types[key] in[int,float]:
                df[key] = df[key].apply(lambda x: x.replace(',', '').strip())
        df = df.astype(column_types)
    if remove_duplicates:
        #Clean Duplicates
        df.drop_duplicates(inplace=True)
    return df

#Get last instance date of each month data
def get_last_instance_date(schemedf,date_column='date_valued'):
    schemedf = schemedf.reset_index()
    #print(schemedf.info())
    schemedf['data_month'] = pd.PeriodIndex(schemedf[date_column].dt.to_period('M'), freq='M').to_timestamp()
    schemedf = schemedf.merge(schemedf.groupby('data_month')[date_column].agg('max'), on = 'data_month', how = 'left')
    columns = {
        date_column+'_x':date_column,
        date_column+'_y':date_column+'_max',
    }    
    schemedf.rename(columns=columns,inplace=True)    
    schemedf = schemedf[schemedf[date_column]==schemedf[date_column+'_max']]
    #print(schemedf)
    schemedf.drop([date_column,date_column+'_max'], axis=1, inplace=True)
    #print(schemedf.info())
    schemedf.set_index('data_month',inplace=True)
    return schemedf

def clean_data_using_sma(schemedata,column_to_clean,period_length=7):
    cleaned_column = column_to_clean+'_clean'
    SMA_column = 'SMA'+str(period_length)
    SMADEV_column = SMA_column+'_DEV'
    STDEV_column = 'ST_DEV'+str(period_length)
    
    schemedata[SMA_column] = schemedata[column_to_clean].astype(float).rolling(period_length).mean()
    schemedata[column_to_clean] = schemedata[column_to_clean].astype(float)
    schemedata[SMADEV_column] = schemedata[column_to_clean] - schemedata[SMA_column]
    schemedata[STDEV_column] = schemedata[SMADEV_column].pow(2).rolling(period_length).apply(lambda x: np.sqrt(x.mean()))
    
    schemedata[cleaned_column] = schemedata.apply(lambda x: x[SMA_column] if abs(x[SMADEV_column]/x[STDEV_column])>1 else x[column_to_clean],axis=1)
    #Drop intermediate columns
    schemedata.drop([SMA_column,SMADEV_column,STDEV_column], axis=1, inplace=True)
    return schemedata


# In[18]:


columntypes = {
    'sale_price_per_unit':float,
    'repurchase_price_per_unit': float,
    'net_asset_value': float,
    'outstanding_number_of_units': float,
    'nav_per_unit': float
}

columns_to_analyse = ['sale_price_per_unit', 'repurchase_price_per_unit']
data = load_data(filepath, list_of_date_columns, indexing_column,column_types=columntypes,remove_duplicates=True)

#Choose a scheme to analyse
def filter_scheme(scheme_id,data):
    schemedata = data[data['scheme_id']==scheme_id].copy()
    return schemedata

def perform_prediction(df,columns_to_analyse,model_to_use='VAR',clean_outliers = False,period_length=15):
    model_dict = {
        'VAR': [VAR_multivariate_forecast, plot_VAR_multivariate_forecast],
        'Random_forest': [RF_multivariate_forecast, plot_RF_multivariate_forecast],
        'ARIMA': [arima_forecast, plot_arima_forecast]
    }
    if clean_outliers:
        #Clean some numerical columns using SMA 15
        for column in columns_to_analyse:
            df = clean_data_using_sma(df,column_to_clean=column,period_length=period_length)
    #update columns to analyse list with clean columns
    columns_to_analyse = [col+'_clean' for col in columns_to_analyse] if clean_outliers else columns_to_analyse
    #Model selection for preferably data prediction
    model_functions = model_dict[model_to_use]
    result = model_functions[0](df,columns_to_analyse)
    rmodel_functions[1](df,result,columns_to_analyse)

# Extract 'Close' column and convert to DataFrame
#plt.plot(schemedata[[col+'_clean' for col in columns_to_analyse]])


# In[20]:


from statsmodels.tsa.vector_ar.var_model import VAR

def VAR_multivariate_forecast(df,column_list):
    df = df.sort_index(ascending=True)
    datta2_ = df[column_list].copy()
    model = VAR(datta2_)
    model_fit = model.fit()
    # Forecast future values
    forecast = model_fit.forecast(model_fit.endog, steps=5)
    return forecast

def plot_VAR_multivariate_forecast(df,forecast,column_list,scheme_name=''):
    df = df.sort_index(ascending=True)
    df_ = df[column_list].copy()
    prediction_dates = pd.date_range(start=df_.index.to_list()[-1], end=df_.index.to_list()[-1]+pd.DateOffset(months=5), freq='MS')
    # Plot the forecasted values
    plt.figure(figsize=(10, 6))
    for i in range(len(df_.columns)):
        plt.plot(df_.index, df_.iloc[:, i], label=df_.columns[i])
        plt.plot(prediction_dates[1:], forecast[:, i], 'r--', label='Forecast '+df_.columns[i])
    plt.legend()
    plt.title(scheme_name+' Multivariate Forecast using VAR')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.show()


def get_forecast(df,clean_outliers,columns_to_analyse,period_length):
    if clean_outliers:
        #Clean some numerical columns using SMA 15
        for i,column in enumerate(columns_to_analyse):
            df = clean_data_using_sma(df,column_to_clean=column,period_length=period_length)
            columns_to_analyse[i] = column +'_clean'
    # Reduce to equal time intervals i.e monthly
    df = get_last_instance_date(df)
    forecast = VAR_multivariate_forecast(df,columns_to_analyse)
    return df, forecast

def make_forecast_df(df,forecast,columns_to_analyse):
    df_ = df.sort_index(ascending=True).copy()
    forecast_df = pd.DataFrame(forecast)
    forecast_df.columns = columns_to_analyse
    forecast_dates = pd.date_range(start=df_.index.to_list()[-1], end=df_.index.to_list()[-1]+pd.DateOffset(months=5), freq='MS')
    forecast_dates_df = pd.DataFrame(forecast_dates,columns=['data_month'])
    forecast_data = forecast_df.merge(forecast_dates_df, how='left',left_index=True,right_index = True)
    forecast_data.set_index('data_month', inplace=True)
    forecast_data['scheme_id'] = list(df['scheme_id'])[0]
    forecast_data['scheme_name'] = list(df['scheme_name'])[0]
    forecast_data['nature'] = 'Forecast'
    return forecast_data

def combine_data(df,forecast_df):
    df['nature'] = 'Actual'
    full_data = pd.concat([df[columns_to_analyse+['scheme_id','scheme_name','nature']],forecast_df], axis=0)
    full_data.sort_index(ascending=False, inplace=True)
    return full_data


# In[22]:


clean_outliers = True
period_length = 15
schemes = [1,2,3,4,5,6]
final_data = []
for scheme in schemes:
    columns_to_analyse = ['sale_price_per_unit', 'repurchase_price_per_unit']
    schemedata = filter_scheme(scheme,data)
    schemedata, forecast = get_forecast(schemedata,clean_outliers,columns_to_analyse,period_length)
    forecast_df = make_forecast_df(schemedata,forecast,columns_to_analyse)
    plot_VAR_multivariate_forecast(schemedata,forecast,columns_to_analyse,scheme_name = list(schemedata['scheme_name'])[0])
    if type(final_data) == list:
        final_data = combine_data(schemedata,forecast_df)
    else:
        final_df = combine_data(schemedata,forecast_df)
        final_data = pd.concat([final_data,final_df], axis=0)


# In[151]:


#print('Export final data to CSV file')
final_data.to_csv("/Users/sherbhanu/Desktop/Anaconda Files /FundData_UTT.csv")
print('Done')


# In[99]:


final_data[final_data.scheme_id==6]


# In[71]:


forecast_df = make_forecast_df(schemedata,forecast,columns_to_analyse)
forecast_df


# In[85]:


final_data=[]
type(final_data) == list


# In[154]:


new_data = pd.concat([full_data,schemedata], axis=0)
new_data.sort_index(ascending=False, inplace=True)
new_data


# In[165]:


from sklearn.ensemble import RandomForestRegressor
#Slipt data into traning and prediction data sets
def RF_split_data(df,feature_list,target,split_at_date):
    df = df.sort_index(ascending=True)
    datta2_ = df[feature_list+[target]].copy()
    feature_training_data = datta2_[datta2_.index < pd.to_datetime(split_at_date)][feature_list].copy()
    target_training_data = datta2_[datta2_.index < pd.to_datetime(split_at_date)][target].copy()
    feature_prediction_data = datta2_[datta2_.index >= pd.to_datetime(split_at_date)][feature_list].copy()
    data = {
        'training':[feature_training_data,pd.DataFrame(target_training_data)],
        'prediction': feature_prediction_data
    }
    return data

def RF_multivariate_forecast(df,feature_list,target,split_at_date):
    data = RF_split_data(df,feature_list,target,split_at_date)
    model = RandomForestRegressor
    model.fit(data['training'][0],data['training'][1])
    predictions = model.predict(data['prediction'])
    return predictions,data

def plot_RF_multivariate_forecast(data,predictions):
    # Plot the predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(data['training'][1].index, data['training'][1], label='Desired Target')
    plt.plot(data['prediction'].index, predictions, 'r--', label='Predicted Target')
    plt.legend()
    plt.title('Multivariate Forecast using Random Forest')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.show()


# In[168]:


feature_list = ['net_asset_value','outstanding_number_of_units','nav_per_unit']
target = 'repurchase_price_per_unit_clean'
split_at_date = '2023-01-01'

#Get preditions
predictions,data = RF_multivariate_forecast(schemedata,feature_list,target,split_at_date)

#Plot the results
plot_RF_multivariate_forecast(data,predictions)


# In[76]:


# Separate the features and target variables
X ,y = data['training'][0],data['training'][1]
# Fit a Random Forest regression model
model = RandomForestRegressor()
model.fit(X, y)
# Generate future data for Random Forest example
future_data = data['prediction']
# Predict future values
predictions = model.predict(future_data)

# Predict future values
# future_data = pd.read_csv('future_data.csv')
#predictions = model.predict(future_data)
# Plot the predicted values
plt.figure(figsize=(10, 6))
plt.plot(schemedata.index, schemedata[target], label='Actual Target')
plt.plot(data['training'][1].index, data['training'][1], label='Desired Target')
plt.plot(data['prediction'].index, predictions, 'r--', label='Predicted Target')
plt.legend()
plt.title('Multivariate Forecast using Random Forest')
plt.xlabel('Time')
plt.ylabel('Values')
plt.show()


# In[100]:


# Importing necessary libraries
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM
import matplotlib.pyplot as plt
# Load multivariate time series data
#data = pd.read_csv('VECM-multivariate_data.csv')

columnlist = ["repurchase_price_per_unit","outstanding_number_of_units"]
#columnlist = ["net_asset_value","outstanding_number_of_units"]

data1 = get_last_instance_date(schemedata)
data1 = data1[columnlist].copy()
data1 = data1.sort_index(ascending=True)

# Fit the VECM model
model = VECM(data1)
model_fit = model.fit()
# Forecast future values
forecast = model_fit.predict(steps=5)
prediction_dates = pd.date_range(start=data1.index.to_list()[-1], end=data1.index.to_list()[-1]+pd.DateOffset(months=5), freq='MS')
# Plot the forecasted values
plt.figure(figsize=(10, 6))
for i in range(len(data1.columns)):
    plt.plot(data1.index, data1.iloc[:, i], label=data1.columns[i])
    plt.plot(prediction_dates[1:], forecast[:, i], 'r--', label='Forecast '+data1.columns[i])
    #plt.plot(range(len(data), len(data) + 5), forecast[:, i], 'r--', label='Forecast '+data.columns[i])
plt.legend()
plt.title('Multivariate Forecast using VECM')
plt.xlabel('Time')
plt.ylabel('Values')
plt.show()


# In[ ]:




