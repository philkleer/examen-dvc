import pandas as pd 
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import json
from pathlib import Path

print(joblib.__version__)
print('Predicting on test data ...')

X_test = pd.read_csv('data/processed_data/normalized/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')
y_test = np.ravel(y_test)

# loading best model parameters
train_model = joblib.load('models/trained_model.pkl') 

# training (again)
y_pred = train_model.predict(X_test)

# model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Test MSE: {mse:.4f}')
print(f'Test R2: {r2:.4f}')

# saving into a file
metrics = {
    'mse': mse,
    'r2' : r2,
    'params': train_model.get_params()
}

# saving metrics
path = Path('./metrics/scores.json')
path.write_text(json.dumps(metrics))
print(f'Scores have been successfully save in {path}.')

# saving predictions
pred_path = Path('./data/predictions.csv')
pred_df = pd.DataFrame(y_pred, columns=['predictions'])
pred_df.to_csv(pred_path, index=False)
print(f'Predictions have been successfully saved in {pred_path}.')