import os

os.system('python src/data_ingestion.py')
print('data ingestion run')

os.system('python src/data_preprocessing.py')
print('data preprocessing run')

os.system('python src/feature_engineering.py')
print('feature engineering run')

os.system('python src/model_building.py')
print('model building run')

os.system('python src/model_evaluation.py')
print('model evaluation run')



