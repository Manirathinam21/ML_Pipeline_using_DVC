import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

#loading the model
model= joblib.load('model.joblib')
test_data= pd.read_csv('./data/features/test_bow.csv')

test_x= test_data.iloc[:,0:-1].values
test_y= test_data.iloc[:,-1].values

y_pred= model.predict(test_x)
y_pred_proba = model.predict_proba(test_x)[:, 1]

# Calculate evaluation metrics
accuracy= accuracy_score(test_y, y_pred)
precision = precision_score(test_y, y_pred)
recall = recall_score(test_y, y_pred)
auc = roc_auc_score(test_y, y_pred_proba)

metrics= {
          'accuracy': accuracy,
          'precision':precision,
          'recall':recall,
          'auc':auc
          }

with open('metrics.json', 'w') as file:
    json.dump(metrics, file, indent=4)



