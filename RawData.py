

import httpx
import numpy as np
import pandas as pd
from parsel import Selector
import time
from datetime import datetime

today = datetime.now().strftime("%d%m%Y")
payload = {
    'draw':'1',
    'columns%5B0%5D%5Bdata%5D':' DT_RowIndex',
    'columns%5B0%5D%5Bname%5D':' DT_RowIndex',
    'columns%5B0%5D%5Bsearchable%5D':' false',
    'columns%5B0%5D%5Borderable%5D':' false',
    'columns%5B0%5D%5Bsearch%5D%5Bvalue%5D':' ',
    'columns%5B0%5D%5Bsearch%5D%5Bregex%5D':' false',
    'columns%5B1%5D%5Bdata%5D':' sname',
    'columns%5B1%5D%5Bname%5D':' sname.name',
    'columns%5B1%5D%5Bsearchable%5D':' true',
    'columns%5B1%5D%5Borderable%5D':' true',
    'columns%5B1%5D%5Bsearch%5D%5Bvalue%5D':' ',
    'columns%5B1%5D%5Bsearch%5D%5Bregex%5D':' false',
    'columns%5B2%5D%5Bdata%5D':' net_asset_value',
    'columns%5B2%5D%5Bname%5D':' net_asset_value',
    'columns%5B2%5D%5Bsearchable%5D':' true',
    'columns%5B2%5D%5Borderable%5D':' true',
    'columns%5B2%5D%5Bsearch%5D%5Bvalue%5D':' ',
    'columns%5B2%5D%5Bsearch%5D%5Bregex%5D':' false',
    'columns%5B3%5D%5Bdata%5D':' outstanding_number_of_units',
    'columns%5B3%5D%5Bname%5D':' outstanding_number_of_units',
    'columns%5B3%5D%5Bsearchable%5D':' true',
    'columns%5B3%5D%5Borderable%5D':' true',
    'columns%5B3%5D%5Bsearch%5D%5Bvalue%5D':' ',
    'columns%5B3%5D%5Bsearch%5D%5Bregex%5D':' false',
    'columns%5B4%5D%5Bdata%5D':' nav_per_unit',
    'columns%5B4%5D%5Bname%5D':' nav_per_unit',
    'columns%5B4%5D%5Bsearchable%5D':' true',
    'columns%5B4%5D%5Borderable%5D':' true',
    'columns%5B4%5D%5Bsearch%5D%5Bvalue%5D':' ',
    'columns%5B4%5D%5Bsearch%5D%5Bregex%5D':' false',
    'columns%5B5%5D%5Bdata%5D':' sale_price_per_unit',
    'columns%5B5%5D%5Bname%5D':' sale_price_per_unit',
    'columns%5B5%5D%5Bsearchable%5D':' true',
    'columns%5B5%5D%5Borderable%5D':' true',
    'columns%5B5%5D%5Bsearch%5D%5Bvalue%5D':' ',
    'columns%5B5%5D%5Bsearch%5D%5Bregex%5D':' false',
    'columns%5B6%5D%5Bdata%5D':' repurchase_price_per_unit',
    'columns%5B6%5D%5Bname%5D':' repurchase_price_per_unit',
    'columns%5B6%5D%5Bsearchable%5D':' true',
    'columns%5B6%5D%5Borderable%5D':' true',
    'columns%5B6%5D%5Bsearch%5D%5Bvalue%5D':' ',
    'columns%5B6%5D%5Bsearch%5D%5Bregex%5D':' false',
    'columns%5B7%5D%5Bdata%5D':' date_valued',
    'columns%5B7%5D%5Bname%5D':' date_valued',
    'columns%5B7%5D%5Bsearchable%5D':' true',
    'columns%5B7%5D%5Borderable%5D':' true',
    'columns%5B7%5D%5Bsearch%5D%5Bvalue%5D':' ',
    'columns%5B7%5D%5Bsearch%5D%5Bregex%5D':' false',
    'start':'0',
    'length':'50',
    'search%5Bvalue%5D':' ',
    'search%5Bregex%5D':' false'

}

endpoint = 'https://www.uttamis.co.tz/navs'
orign_link = 'https://www.uttamis.co.tz/fund-performance'

headers = {
    # Lets use Chrome browser on Windows:
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
}

final_df = pd.DataFrame()

with httpx.Client(headers=headers, timeout=30.0) as session:
    response = session.get(orign_link)
    tree = Selector(response.text)
    csfr_token = tree.xpath('//meta[contains(@name,"csrf-token")]/@content').get()
    payload['csfr_token'] = csfr_token

    post_header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
        'Accept':'application/json, text/javascript, */*; q=0.01',
        'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
        'Origin':'https://www.uttamis.co.tz',
        'X-CSRF-TOKEN':csfr_token,
        'X-Requested-With':'XMLHttpRequest',
        'sec-ch-ua':'"Not/A)Brand";v="8", "Chromium";v="126", "Microsoft Edge";v="126"',
        'sec-ch-ua-mobile':'?0',
        'sec-ch-ua-platform':'"Windows"'}
    
    # Set initial start and length values
    start = 0
    length = 50
    total_records = float('inf')  # Placeholder for total records
    payload['start'] = str(start)
    payload['length'] = str(length)
    
    while start < total_records:
        try:
            post_response = session.post(endpoint, data=payload, headers=post_header)
            post_response.raise_for_status()  # Raise an error for bad status codes
            post_response_json = post_response.json()
            
            # Update total records
            if total_records == float('inf'):
                total_records = post_response_json['recordsTotal']
    
            dframe = pd.DataFrame.from_dict(post_response_json['data'])
            final_df = pd.concat([final_df, dframe], ignore_index=True)
            
            # Update the start for the next iteration
            start += length
            payload['start'] = str(start)
            
            print(f'Retrieved {start} of {total_records} records')
        
        except httpx.ReadTimeout:
            print(f"Timeout occurred for start={start}, retrying...")
            time.sleep(5)  # Wait before retrying
            continue  # Retry the same request
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

# Now final_df contains all the data from all pages
print(final_df)
csv_file_path = 'MyData_fund.csv'
final_df.to_csv(csv_file_path, index=False)

#Save working data
csv_file_path = f'Desiredata {today}.csv'
working_columns = ['date_valued','entry_id','scheme_id','net_asset_value','outstanding_number_of_units','nav_per_unit','sale_price_per_unit','repurchase_price_per_unit','scheme_name']
final_df[working_columns].to_csv(csv_file_path, index=False)


# In[18]:


final_df = final_df.sort_index(ascending=True)


# In[11]:


from datetime import datetime

today = datetime.now().strftime("%d%m%Y")
#Save working data
csv_file_path = f'Desiredata {today}.csv'
working_columns = ['date_valued','entry_id','scheme_id','net_asset_value','outstanding_number_of_units','nav_per_unit','sale_price_per_unit','repurchase_price_per_unit','scheme_name']
final_df[working_columns].to_csv(csv_file_path, index=False)


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')

#Loading data in the notebook
df = pd.read_csv('MyData_fund.csv')

#EDA/Preprocessing
print(df)

print(df.head())

print(df.describe())

print(df.info())

df.shape

print(df.columns.tolist())

print(df.isnull().sum())

print(df.nunique())

print(df.duplicated().sum())

print(df.tail())

#Statistical Analysis 
mean_values = df.mean()
print("Mean:\n", mean_values)

median_values = df.median()
print("Median:\n", median_values)

mode_values = df.mode().iloc[0]  
print("Mode:\n", mode_values)


# In[16]:


import matplotlib.pyplot as plt 

numerical_columns = df.select_dtypes(include=['number', 'datetime']).columns
if len(numerical_columns) > 0:
    # Plot histograms for numerical columns
    df[numerical_columns].hist(bins=10, figsize=(12, 8))
    plt.show()
else:
    print("No numerical or datetime columns to plot histograms.")
numerical_columns = ['net_asset_value', 'outstanding_number_of_units', 'nav_per_unit', 'sale_price_per_unit', 'repurchase_price_per_unit']
df[numerical_columns].hist(bins=50, figsize=(20, 15))
plt.show()


# In[30]:


numerical_columns = df.select_dtypes(include=['number', 'datetime']).columns
if len(numerical_columns) > 0:
    # Plot histograms for numerical columns
    plt.figure(figsize=(20, 15))
    df[numerical_columns].plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
    plt.show()
else:
    print("No numerical or datetime columns to plot histograms.")


# In[126]:


numerical_columns = df.select_dtypes(include=['number']).columns
sns.pairplot(df[numerical_columns])
plt.show()


# In[63]:





# In[47]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('MyData_fund.csv', parse_dates=['date_valued', 'approved_at', 'created_at', 'updated_at', 'fmonth', 'dmonth'],infer_datetime_format=True,index_col='date_valued')

list_of_columns = ['net_asset_value','outstanding_number_of_units','nav_per_unit','sale_price_per_unit','repurchase_price_per_unit']

df[list_of_columns] = df[list_of_columns].apply(lambda x:x.str.replace(',','').astype(float))#.apply(remove_commas(list_of_columns), axis=1)
csv_file_path = 'FundData.csv'
df.to_csv(csv_file_path)



# In[48]:


#df = df.reindex()
df.head()


# In[2]:


get_ipython().system('pip install scikit-learn')


# In[71]:


import sklearn as sk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

df = pd.read_csv('FundData.csv', parse_dates=['date_valued', 'approved_at', 'created_at', 'updated_at', 'fmonth', 'dmonth'])#,index_col='date_valued')
df1 = df[["repurchase_price_per_unit","net_asset_value","outstanding_number_of_units","nav_per_unit","sale_price_per_unit","id","entry_id","scheme_id","created_by","is_approved","approved_by"]].copy()

X_std = StandardScaler().fit_transform(df1)
print(X_std)


# In[2]:


import sklearn as sk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('FundData.csv', parse_dates=['date_valued', 'approved_at', 'created_at', 'updated_at', 'fmonth', 'dmonth'])#,index_col='date_valued')
df1 = df[["repurchase_price_per_unit","net_asset_value","outstanding_number_of_units","nav_per_unit","sale_price_per_unit","id","entry_id","scheme_id","created_by","is_approved","approved_by"]].copy()
X_std = StandardScaler().fit_transform(df1)

pca_data = PCA(n_components=3)
pca_model = pca_data.fit_transform(X_std)

reduced_df = pd.DataFrame(data=pca_model, columns=['PCA 1', 'PCA 2', 'PCA 3'])
final_df = pd.concat([reduced_df, df1], axis=1)

targets = final_df['scheme_id'].unique()
colors = ['y', 'g', 'b', 'r']

plt.figure(figsize=(8, 6))
for target, color in zip(targets, colors):
    idx = final_df['scheme_id'] == target
    plt.scatter(final_df.loc[idx, 'PCA 1'], final_df.loc[idx, 'PCA 2'], c=color, s=50)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on UTT Dataset')
plt.show()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# get the correlation matrix with pandas
#correlation = df1['net_asset_value'].corr(df1['repurchase_price_per_unit'])
#print(correlation)

# Calculate the correlation matrix
correlation_matrix = df1[['net_asset_value', 'repurchase_price_per_unit']].corr()

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()


# In[20]:


# Import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset from a csv file
df = pd.read_csv("FundData.csv")

# Define the dependent and independent variables
y = df["nav_per_unit"] # dependent variable
X = df["repurchase_price_per_unit"] # independent variable

# Reshape the variables to fit the model
y = y.values.reshape(-1, 1)
X = X.values.reshape(-1, 1)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Get the model parameters
slope = model.coef_[0][0]
intercept = model.intercept_[0]
r_squared = model.score(X, y)

# Print the model parameters
print(f"slope = {slope:.4f}")
print(f"intercept = {intercept:.4f}")
print(f"r_squared = {r_squared:.4f}")

# Plot the data points and the regression line
sns.regplot(x=X, y=y, ci=None)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear regression of y on x")
plt.show()

