import pandas as pd 
from sklearn.linear_model import Ridge
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import json
from pathlib import Path

print(joblib.__version__)
print('Predicting on test data ...')

X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')
y_test = np.ravel(y_test)

# loading best model parameters
best_model = joblib.load('models/trained_model.joblib') 

# training (again)
y_pred = best_model.predict(X_test)

# model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Test MSE: {mse:.4f}')
print(f'Test R2: {r2:.4f}')

# saving into a file
metrics = {
    'mse': mse,
    'r2' : r2,
    'params': best_model.get_params()
}

path = Path('./metrics/scores.json')
path.write_text(json.dumps(metrics))
print(f'Scores have been successfully save in {path}.')
