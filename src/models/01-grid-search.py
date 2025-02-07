import pandas as pd 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np

print(joblib.__version__)
print('Doing Grid Search ... ')

X_train = pd.read_csv('data/processed_data/normalized/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/splitted/y_train.csv')
y_train = np.ravel(y_train)

# Define model
ridge = Ridge()

# Define parameter grid
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100]  # Small range for efficiency
}

# Grid search
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = './models/best_params.pkl'

# Save best model
joblib.dump(grid_search.best_estimator_.get_params(), model_filename)

print('Parameters of best model saved successfully.')
