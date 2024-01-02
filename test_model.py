import mlflow.sklearn
import pandas as pd
from preprocess import preprocess_data

# Load the registered model from the MLflow Model Registry
model_name = 'PredictiveMaintenanceModel'
model_version = 'Production'  # Use the appropriate version
model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

# Load live data (replace 'path_to_live_data.csv' with the actual path)
live_data = pd.read_csv('path_to_live_data.csv')

# Preprocess the live data if needed (use the same preprocessing used during training)
#X_processed, y = preprocess_data(file_path)

# Make predictions on the live data
live_predictions = model.predict(live_data)

# You can print or further process the predictions as needed
print("Live Predictions:", live_predictions)
