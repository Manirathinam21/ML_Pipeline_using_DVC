import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# fetch the data from data/processed
train_data= pd.read_csv('./data/preprocessed/train_preprocessed.csv') 
test_data= pd.read_csv('./data/preprocessed/test_preprocessed.csv') 

train_data.fillna('',inplace=True)
test_data.fillna('',inplace=True)

train_x= train_data['content'].values
train_y= train_data['sentiment'].values

test_x= test_data['content'].values
test_y= test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer= CountVectorizer(max_features=50)

# Fit the vectorizer on the training data and transform it
train_x_bow= vectorizer.fit_transform(train_x)

# Transform the test data using the same vectorizer
test_x_bow= vectorizer.transform(test_x)

train_df= pd.DataFrame(train_x_bow.toarray())
train_df['label']= train_y

test_df= pd.DataFrame(test_x_bow.toarray())
test_df['label']= test_y

# store the data inside data/features
data_path= os.path.join('data','features')
os.makedirs(data_path, exist_ok=True)

train_df.to_csv(os.path.join(data_path,'train_bow.csv'))
test_df.to_csv(os.path.join(data_path,'test_bow.csv'))

