#Importing Libraries
import pandas as pd
import numpy as np
from scipy.stats import kruskal
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pmd

#Data Preparation
data=pd.read_csv('Demands.csv')
data.set_index('time', inplace=True)
data.index = pd.to_datetime(data.index)
print(data.head())

#Splitting Data into Train and Test Sets
train_data = data[data.index < '2018-02-01 00:00:00']
test_data = data[(data.index >= '2018-02-01 00:00:00') & (data.index <= '2018-02-26 23:00:00')]

#Kruskal Wallis Test For Seasonality
load_data=data['load']

#Creating an empty list of lists to store the spliited lists needed for the test
daily_demand = []

#Iterating over the 30 days to split each 24 observation of each day
for i in range(30):
  start_index = i*24
  end_index = (i+1)*24
  daily_demand.append(load_data[start_index:end_index])
#Giving kruskal 30 samples to check seasonality
SeasonalityTest = kruskal(*daily_demand)
p_kruskal=SeasonalityTest[1]
print('p_kruskal=', p_kruskal)

#Augmented Dickey Fuller Test For Stationarity
StationarityTest = adfuller(data.load, autolag='AIC')
p_ADF = StationarityTest[1]
print('p_ADF=', StationarityTest[1])


#Correlation values & Heatmap

#Getting the correlation between load and all variables available in the dataset
correlations=data.corr()['load']

#Dropping the correlation of the load with itself '=1'
correlations=correlations.drop('load')
print(correlations)

#Getting the index with the max absolute value
max_corr_variable=correlations.abs().idxmax()
print('max_corr_variable=',max_corr_variable)

#Accesing the max correlation value from the max absolute value index
max_corr_value=correlations.loc[max_corr_variable]
print('max_corr_value=',max_corr_value)

#Creating the correlation heatmap
fig,ax=plt.subplots(figsize=(15,5))
heatmap = sns.heatmap(data.corr(), cmap="Blues", vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 20}, pad=20)
plt.show()

Y = pd.DataFrame(train_data['load'])
X = pd.DataFrame(train_data[max_corr_variable])
Xtest = pd.DataFrame(test_data[max_corr_variable])

def get_parameters(model):
  parameters=model.get_params()
  order=parameters.get('order')
  seasonal_order=parameters.get('seasonal_order')
  p,d,q,P,D,Q,m=[order[0],order[1],order[2],seasonal_order[0],seasonal_order[1],seasonal_order[2],seasonal_order[3]]
  return p,d,q,P,D,Q,m


if p_ADF <= 0.05:
  # stationary data
  if p_kruskal > 0.05:  # no seasonality
    if max_corr_value > 0.5:  # strong correlation

      # ARIMAX
      autoarima_model = pmd.auto_arima(Y, X=X, start_p=0,
                                       start_q=1, max_order=5,
                                       error_action='ignore', suppress_warnings=True, stepwise=True, test='adf',
                                       seasonal=False, trace=True)
      [p, d, q, P, D, Q, m] = get_parameters(autoarima_model)
      model = SARIMAX(Y, X=X, order=(p, d, q), exog=X, seasonal_order=(P, D, Q, m))
      predicted_loads = model.fit().forecast(exog=Xtest,steps=624)

      # ARIMAX using SARIMAX function with seasonal order(0,0,0,0) & exog=train_data[max_corr_variable]
    else:   # weak correlation
      # ARIMA
      autoarima_model = pmd.auto_arima(Y,start_p = 0, start_q = 0, max_order = 5,error_action='ignore', suppress_warnings=True, stepwise=True, test='adf', seasonal=False, trace=True)
      [p, d, q, P, D, Q, m] = get_parameters(autoarima_model)     # ARIMA with order(p,d,q)
      model = SARIMAX(Y, order=(p, d, q), exog=X, seasonal_order=(P, D, Q, m))
      predicted_loads = model.fit().forecast(steps=624)
  else:  # seasonal
    if max_corr_value > 0.5:    # strong correlation
      # SARIMAX
      autoarima_model = pmd.auto_arima(Y,start_p = 0, start_q = 0, max_order = 5,error_action='ignore', suppress_warnings=True, stepwise=True, test='adf', m=24, seasonal=True, trace=True)
      [p, d, q, P, D, Q, m] = get_parameters(autoarima_model)
      model = SARIMAX(Y, X=X, order=(p,d,q), exog=X, seasonal_order=(P,D,Q,m))
      predicted_loads = model.fit().forecast(exog=Xtest, steps=624)
    else:     # weak correlation
      # SARIMA
      autoarima_model = pmd.auto_arima(Y,start_p = 0, start_q = 0, max_order = 5,error_action='ignore', suppress_warnings=True, stepwise=True, test='adf', m=24, seasonal=True, trace=True)
      [p, d, q, P, D, Q, m] = get_parameters(autoarima_model)
      model = SARIMAX(Y, order=(p,d,q), seasonal_order=(P,D,Q,m))
      predicted_loads = model.fit().forecast(steps=624)
else:   # non stationary data
  if max_corr_value > 0.5:    # strong correlation
    # SARIMAX
    autoarima_model = pmd.auto_arima(Y, X=X,start_p = 0, start_q = 0, max_order = 5,error_action='ignore', suppress_warnings=True, stepwise=True, test='adf', m=24, seasonal=True, trace=True)
    [p, d, q, P, D, Q, m] = get_parameters(autoarima_model)
    model = SARIMAX(Y, order=(p,d,q), exog=X, seasonal_order=(P,D,Q,m))
    predicted_loads = model.fit().forecast(exog=Xtest, steps=624)
  else:    # weak correlation
    # SARIMA
    autoarima_model = pmd.auto_arima(Y, start_p=0, start_q=0, max_order=5,
                                     error_action='ignore', suppress_warnings=True, stepwise=True, test='adf', m=24,
                                     seasonal=True, trace=True)
    [p, d, q, P, D, Q, m] = get_parameters(autoarima_model)
    model = SARIMAX(Y, order=(p,d,q), seasonal_order=(P,D,Q,m))
    predicted_loads = model.fit().forecast(steps=624)

plt.plot(predicted_loads, label='predicted')
plt.plot(test_data['load'], label='actual')
plt.legend()
plt.show()

#Calculate MAPE
def mape(actual, pred):
  actual, pred = np.array(actual), np.array(pred)
  return np.mean(np.abs((actual - pred) / actual)) * 100

print(mape(test_data['load'], predicted_loads))