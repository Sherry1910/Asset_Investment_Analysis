#!/usr/bin/env python
# coding: utf-8

# In[38]:


# import python pandas library
import pandas as pd
import matplotlib.pyplot as plt

# read the dataset using pandas read_csv() 
# function
datta = pd.read_csv("/Users/sherbhanu/Desktop/Anaconda Files /Desiredata.csv",parse_dates=['date_valued'],index_col='date_valued',header=0)
datta1 = datta[["repurchase_price_per_unit","net_asset_value","outstanding_number_of_units","nav_per_unit","sale_price_per_unit","entry_id","scheme_id"]].copy()
plt.plot(datta1)


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

umojadf = datta[datta['scheme_id']==1].copy()
datta2 = umojadf[['sale_price_per_unit','SMA30']].copy()
datta2 = datta2['sale_price_per_unit_clean'].astype(float)

# Split the data into train and test sets based on date
datta2 = datta2.sort_index(ascending=True)
train_umoja = datta2[datta2.index < pd.to_datetime("2021-01-01")]['sale_price_per_unit_clean'].copy()
test_umoja = datta2[datta2.index >= pd.to_datetime("2021-01-01")]['sale_price_per_unit_clean'].copy()
# Fit an ARIMA model (ARMA is a special case of ARIMA)
# You might need to experiment with different orders (p, d, q)
ARIMAmodel = ARIMA(train_umoja, order=(2,2,1))
ARIMAmodel_fit = ARIMAmodel.fit()

# Make forecasts
y_pred = ARIMAmodel_fit.get_forecast(steps=len(test_umoja.index))
y_pred_df = y_pred.conf_int()
y_pred_df['Predictions'] = y_pred.prediction_results.forecasts[0]
y_pred_df.index = test_umoja.index

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train_umoja.index, train_umoja, color="black", label='Train')
plt.plot(test_umoja.index, test_umoja, color="red", label='Test')
plt.plot(y_pred_df.index, y_pred_df['Predictions'], color="green", label='Predictions')
plt.ylabel('Sale Price Per Unit')
plt.xlabel('Date Valued')
plt.xticks(rotation=30)
plt.title("Train/Test Split with Predictions for UTT Data")
plt.legend()
plt.show()


# In[8]:


print(datta.head(10))
values = datta1.values
parts = int(len(values)/3)
p1, p2, p3 = values[0:parts], values[parts:(
    parts*2)], values[(parts*2):(parts*3)]
m1, m2, m3 = p1.mean(), p2.mean(), p3.mean()
v1, v2, v3 = p1.var(), p2.var(), p3.var()
print('mean1=%f, mean2=%f, mean2=%f' % (m1, m2, m3))
print('variance1=%f, variance2=%f, variance2=%f' % (v1, v2, v3))


# In[5]:


import math 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datta1 = datta[["repurchase_price_per_unit","net_asset_value","outstanding_number_of_units","nav_per_unit","sale_price_per_unit","entry_id","scheme_id"]].copy()
values = np.log(datta1.values)

print(values[:,5])
plt.plot(values)


# In[51]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filepath = "/Users/sherbhanu/Desktop/Anaconda Files /Desiredata.csv"
column_list = ["repurchase_price_per_unit","net_asset_value","outstanding_number_of_units","nav_per_unit","sale_price_per_unit","entry_id","scheme_id"]
list_of_date_columns = ['date_valued']
indexing_column = ['date_valued']

def load_data(filepath, list_of_date_columns, indexing_column,column_types={},remove_duplicates=False):
    df = pd.read_csv(filepath, parse_dates=list_of_date_columns,index_col=indexing_column)
    if column_types:
        df = df.astype(column_types)
    if remove_duplicates:
        #Clean Duplicates
        df.drop_duplicates(inplace=True)
    return df

#Get last instance date of each month data
def get_last_instance_date(schemedf):
    schemedf = schemedf.reset_index()
    #print(schemedf.info())
    schemedf['data_month'] = pd.PeriodIndex(schemedf.date_valued.dt.to_period('M'), freq='M').to_timestamp()
    schemedf = schemedf.merge(schemedf.groupby('data_month').date_valued.agg('max'), on = 'data_month', how = 'left')
    columns = {
        'date_valued_x':'date_valued',
        'date_valued_y':'date_valued_max',
    }    
    schemedf.rename(columns=columns,inplace=True)    
    schemedf = schemedf[schemedf.date_valued==schemedf.date_valued_max]
    #print(schemedf)
    schemedf.drop(['date_valued','date_valued_max'], axis=1, inplace=True)
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

columntypes = {
    'sale_price_per_unit':float,
    'repurchase_price_per_unit': float
}
data = load_data(filepath, list_of_date_columns, indexing_column,column_types=columntypes,remove_duplicates=True)

#Choose a scheme to analyse
schemedata = data[data['scheme_id']==1].copy()

#Clean some numerical columns using SMA 15
columns_to_analyse = ['sale_price_per_unit', 'repurchase_price_per_unit']
for column in columns_to_analyse:
    schemedata = clean_data_using_sma(schemedata,column_to_clean=column,period_length=15)

# Reduce to equal time intervals i.e monthly
#umojadf = get_last_instance_date(umojadf)

# Extract 'Close' column and convert to DataFrame
plt.plot(schemedata[[col+'_clean' for col in columns_to_analyse]])



# In[217]:


# import python packages
import pandas as pd
from statsmodels.tsa.stattools import adfuller
datta2['sale_price_per_unit_clean'] = datta2['sale_price_per_unit_clean'].astype(float)
# extracting only the passengers count using values function
values = datta2['sale_price_per_unit_clean'].to_list()#.values

# passing the extracted passengers count to adfuller function.
# result of adfuller function is stored in a res variable
res = adfuller(values)

# Printing the statistical result of the adfuller test
print('Augmneted Dickey_fuller Statistic: %f' % res[0])
print('p-value: %f' % res[1])

# printing the critical values at different alpha levels.
print('critical values at different levels:')
for k, v in res[4].items():
	print('\t%s: %.3f' % (k, v))


# In[268]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Split the data into train and test sets based on date
datta2 = datta2.sort_index(ascending=True)
train_umoja = datta2[datta2.index < pd.to_datetime("2021-01-01")]['sale_price_per_unit_clean'].copy()
test_umoja = datta2[datta2.index >= pd.to_datetime("2021-01-01")]['sale_price_per_unit_clean'].copy()
# Fit an ARIMA model (ARMA is a special case of ARIMA)
# You might need to experiment with different orders (p, d, q)
ARIMAmodel = ARIMA(train_umoja, order=(2,0,2))
ARIMAmodel_fit = ARIMAmodel.fit()

# Make forecasts
y_pred = ARIMAmodel_fit.get_forecast(steps=len(test_umoja.index))
y_pred_df = y_pred.conf_int()
y_pred_df['Predictions'] = y_pred.prediction_results.forecasts[0]
y_pred_df.index = test_umoja.index

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train_umoja.index, train_umoja, color="black", label='Train')
plt.plot(test_umoja.index, test_umoja, color="red", label='Test')
plt.plot(y_pred_df.index, y_pred_df['Predictions'], color="green", label='Predictions')
plt.ylabel('Sale Price Per Unit')
plt.xlabel('Date Valued')
plt.xticks(rotation=30)
plt.title("Train/Test Split with Predictions for UTT Data")
plt.legend()
plt.show()

#Calculate MSE
mae = mean_absolute_error(test_umoja, y_pred_df['Predictions'])
print(f"Mean Absolute Error (MAE): {mae:.2f}")


# In[264]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(datta2['sale_price_per_unit_clean'].diff().dropna());


# In[265]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(datta2['sale_price_per_unit_clean'].diff().dropna());


# In[256]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

#datta2 = datta2['sale_price_per_unit_clean'].astype(float)

# Split the data into train and test sets based on date
schemedata = schemedata.sort_index(ascending=True)
train_umoja = datta2[datta2.index < pd.to_datetime("2021-01-01")]['sale_price_per_unit'].copy()
test_umoja = datta2[datta2.index >= pd.to_datetime("2021-01-01")]['sale_price_per_unit'].copy()
# Fit an ARIMA model (ARMA is a special case of ARIMA)
# You might need to experiment with different orders (p, d, q)
ARIMAmodel = ARIMA(train_umoja, order=(2,2,1))
ARIMAmodel_fit = ARIMAmodel.fit()

# Make forecasts
y_pred = ARIMAmodel_fit.get_forecast(steps=len(test_umoja.index))
y_pred_df = y_pred.conf_int()
y_pred_df['Predictions'] = y_pred.prediction_results.forecasts[0]
y_pred_df.index = test_umoja.index

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train_umoja.index, train_umoja, color="black", label='Train')
plt.plot(test_umoja.index, test_umoja, color="red", label='Test')
plt.plot(y_pred_df.index, y_pred_df['Predictions'], color="green", label='Predictions')
plt.ylabel('Sale Price Per Unit')
plt.xlabel('Date Valued')
plt.xticks(rotation=30)
plt.title("Train/Test Split with Predictions for UTT Data")
plt.legend()
plt.show()


# In[54]:


import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
import matplotlib.pyplot as plt

schemedata = schemedata.sort_index(ascending=True)
datta2_ = schemedata[['sale_price_per_unit_clean', 'repurchase_price_per_unit_clean']]
model = VAR(datta2_)
model_fit = model.fit()
# Forecast future values
forecast = model_fit.forecast(model_fit.endog, steps=5)
prediction_dates = pd.date_range(start=datta2_.index.to_list()[-1], end=datta2_.index.to_list()[-1]+pd.DateOffset(months=5), freq='MS')
# Plot the forecasted values
plt.figure(figsize=(10, 6))
for i in range(len(datta2_.columns)):
    plt.plot(datta2_.index, datta2_.iloc[:, i], label=datta2_.columns[i])
    plt.plot(prediction_dates, forecast[:, i], 'r--', label='Forecast '+datta2_.columns[i])
plt.legend()
plt.title('Multivariate Forecast using VAR')
plt.xlabel('Time')
plt.ylabel('Values')
plt.show()


# In[53]:


prediction_dates = pd.date_range(start=datta2_.index.to_list()[-1], end=datta2_.index.to_list()[-1]+pd.DateOffset(months=5), freq='MS')   
print(prediction_dates)


# In[48]:


datta3_=datta.copy()
datta3_=datta3_.sort_index(ascending=True)
datta3_[datta3_.scheme_id==1].index.to_list()[-1]


# In[49]:


datta3__=datta3_.drop_duplicates()
datta3__[datta3__.scheme_id==1].index.to_list()[-1]


# In[ ]:


datta3__[datta3__.scheme_id==1].index.to_list()[-1]
get_last_instance_date(schemedf)


# In[55]:


schemedata


# In[ ]:




