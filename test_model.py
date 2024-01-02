import mlflow.sklearn
import pandas as pd
from preprocess import preprocess_data

# Assuming the live data is in a CSV file (replace 'path_to_live_data.csv' with the actual path)
live_data_path = 'path_to_live_data.csv'
live_data = pd.read_csv(live_data_path)

# Preprocess the live data using the same preprocessing steps as during training
live_data_processed, _ = preprocess_data(live_data_path)

# Load the registered model from the MLflow Model Registry
model_name = 'PredictiveMaintenanceModel'
model_version = 'Production'  # Use the appropriate version
model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

# Make predictions on the live data
live_predictions = model.predict(live_data_processed)

# You can print or further process the predictions as needed
print("Live Predictions:", live_predictions)
