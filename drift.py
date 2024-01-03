import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import os

# Function to monitor drift
def monitor_drift(X_val, y_val, model):
    # Make predictions on the "current validation" set
    y_pred_val = model.predict(X_val)
    # Calculate Mean Squared Error (MSE) as a drift metric
    mse_val = mean_squared_error(y_val, y_pred_val)

    return mse_val

# Function to check for concept drift
def check_concept_drift(X_val, y_val, model, baseline_mse, drift_threshold):
    current_mse = monitor_drift(X_val, y_val, model)
    # Calculate the relative change in MSE
    relative_change = (current_mse - baseline_mse) / baseline_mse

    return relative_change > drift_threshold

# Function to retrain the model
def retrain_model(X_train, y_train, model):
    # Retrain the model on the full training set
    model.fit(X_train, y_train)

    return model

if __name__ == "__main__":
    # Load the preprocessed data
    file_path = 'dummy_sensor_data.csv'
    X_train, y_train = preprocess_data(file_path)

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Get the latest run ID
    latest_run_id = mlflow.search_runs(order_by=['attributes.start_time desc']).iloc[0]['run_id']

    # Construct the model URI
    latest_best_model_path = None
    latest_run_id = None
    mlruns_path = os.path.join(os.getcwd(), 'mlruns')
    for root, dirs, files in os.walk(mlruns_path):
        for dir_name in dirs:
            run_path = os.path.join(root, dir_name)
            artifact_path = os.path.join(run_path, 'artifacts', 'best_model')

            if os.path.exists(artifact_path):
                if latest_run_id is None or dir_name > latest_run_id:
                    latest_run_id = dir_name
                    latest_best_model_path = artifact_path

    if latest_best_model_path:
        model_uri = f"file://{latest_best_model_path}"

    # Load the best model from the MLflow Model Registry
    model = mlflow.sklearn.load_model(model_uri)

    # Monitor drift using a subset of the training data as the "current validation" set
    baseline_mse = monitor_drift(X_train, y_train, model)
    # Define drift monitoring parameters
    drift_threshold = 1

    # Check for concept drift
    if check_concept_drift(X_val, y_val, model, baseline_mse, drift_threshold):
        print("True")
        
    else:
        print("False")
    
