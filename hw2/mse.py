import pandas as pd 
from sklearn.metrics import mean_squared_error

df = pd.read_csv('ml-latest-small/response.csv')
tdf = pd.read_csv('ml-latest-small/ratings_test_truth.csv')
y_true = tdf['rating']
y_pred = df['rating']

print(mean_squared_error(y_true, y_pred))