import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


s_sales = pd.read_csv('train.csv')
s_sales.head(10)

s_sales =s_sales.drop(columns=['store','item'],axis = 1)
s_sales['date']=pd.to_datetime(s_sales['date'])
s_sales['date'] = s_sales['date'].dt.to_period('M')
monthly_sales=s_sales.groupby('date').sum().reset_index()
monthly_sales['date']=monthly_sales['date'].dt.to_timestamp()
#monthly_sales.head(10)

monthly_sales['sales_diff']=monthly_sales['sales'].diff()
monthly_sales =monthly_sales.dropna()
#Preparing the supervised data (recovering 2 month sales)
s_data = monthly_sales.drop(['date', 'sales'], axis=1)

for i in range(1,13):
  col_name= 'month_'+str(i)
  s_data[col_name]=s_data['sales_diff'].shift(i)
  
s_data = s_data.dropna().reset_index(drop=True)


#supervised_data.head(10)

#splitting the data

train_data=s_data[:-12]
test_data=s_data[-12:]
#print(train_data.shape)
#print(test_data.shape)

scaler= MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data=scaler.transform(train_data)
test_data=scaler.transform(test_data)
X_train , y_train = train_data[:,1:], train_data[:,0:1]
X_test , y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
y_test =y_test.ravel()


s_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(s_dates)

act_sales = monthly_sales['sales'][-13:].to_list()


model =LinearRegression()
model.fit(X_train,y_train)

predict = model.predict(X_test)

predict = predict.reshape(-1,1)
test_set = np.concatenate([predict,X_test],axis = 1)
test_set = scaler.inverse_transform(test_set)

  
result_list = []
for index in range ( 0, len(test_set)):
  result_list.append(test_set[index][0] + act_sales[index])
series = pd.Series(result_list, name='Linear prediction')

predict_df['Linear prediction'] = series


#mse = np.sqrt(mean_squared_error(predict_df))
print(predict_df)



