import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from preprocess import preprocess_data

mlflow.set_tracking_uri("mlruns")  # Set to your desired tracking URI
mlflow.sklearn.autolog()

# Load preprocessed data
file_path = 'dummy_sensor_data.csv'
X_processed, y = preprocess_data(file_path)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Save the preprocessed data
#pd.DataFrame(X_train).to_csv('preprocessed_X_train.csv', index=False)
#pd.DataFrame(X_val).to_csv('preprocessed_X_val.csv', index=False)
#pd.DataFrame(y_train).to_csv('preprocessed_y_train.csv', index=False)
#pd.DataFrame(y_val).to_csv('preprocessed_y_val.csv', index=False)

# Create and train the model
model = RandomForestRegressor()

# Define hyperparameters for tuning
param_grid = {
  'n_estimators': [50, 100, 150],
  'max_depth': [None, 10, 20, 30]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(X_train, y_train)

# Get the best model from hyperparameter tuning
best_model = grid_search.best_estimator_

# Log model parameters and metrics with MLflow
with mlflow.start_run() as run:
  # Log preprocessing parameters
  mlflow.log_params({'data_preprocessing': 'ColumnTransformer with OneHotEncoder'})

  # Log best model parameters
  mlflow.log_params({'n_estimators': best_model.n_estimators, 'max_depth': best_model.max_depth})

  # Log training metrics for the best model
  y_pred_train = best_model.predict(X_train)
  mse_train = mean_squared_error(y_train, y_pred_train)
  mlflow.log_metrics({'mse_train': mse_train})

  # Save the best model
  mlflow.sklearn.log_model(best_model, "best_model")

# Register the best model in the MLflow Model Registry
model_name = 'PredictiveMaintenanceModel'
model_stage = 'Production'

# Get the model URI
model_uri = f"best_model"

# Register the best model in the Model Registry
mlflow.register_model(model_uri, model_name)
