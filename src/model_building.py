import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier

# fetch the data from data/feature
train_data= pd.read_csv('./data/features/train_bow.csv')
test_data= pd.read_csv('./data/features/test_bow.csv')

train_x= train_data.iloc[:, 0:-1]
train_y= train_data.iloc[:,-1]

model= GradientBoostingClassifier(n_estimators=50)
model.fit(train_x, train_y)

# save the model using joblib
file_name= 'model.joblib'
joblib.dump(model, file_name)