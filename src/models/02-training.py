import pandas as pd 
from sklearn.linear_model import Ridge
import joblib
import numpy as np

print(joblib.__version__)
print('Training model ... ')

X_train = pd.read_csv('data/processed_data/normalized/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/splitted/y_train.csv')
y_train = np.ravel(y_train)

# loading best model parameters
best_params = joblib.load('models/best_params.pkl') 

# training (again)
best_model = Ridge(**best_params)  # Initialize model with best params
best_model.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = './models/trained_model.pkl'
joblib.dump(best_model, model_filename)
print('Model trained and saved successfully.')
