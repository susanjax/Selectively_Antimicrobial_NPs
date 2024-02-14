import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from V4_ZOI.Models import V4_transform_ZOI
from sklearn.model_selection import train_test_split

processed_data = pd.read_csv(r'C:\Users\user\Desktop\Valya\V4_ZOI\data\preprocessed\final_preprocessed_ZOI.csv')
# print(processed_data.columns)
processed_data = processed_data.drop(['Unnamed: 0','source','reference',], axis=1)
processed_data = processed_data.drop_duplicates()
print(processed_data.info())
X = processed_data.drop(['zoi_np'], axis=1)
Y = processed_data[['zoi_np']].copy()

Xt = V4_transform_ZOI.first_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(Xt, Y, test_size=0.2, random_state = 42)
clf = LazyRegressor(verbose=3,ignore_warnings=True, custom_metric=None)
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[:-0, :]
train.to_csv('Model_comparision_processed_ZOI.csv')
