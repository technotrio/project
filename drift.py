import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Function to monitor drift
def monitor_drift(X_val, y_val, model):
    # Make predictions on the "current validation" set
    y_pred_val = model.predict(X_val)
    print("train:")
    print(y_train[1])
    print("pred:")
    print(y_pred_val[1])
    # Calculate Mean Squared Error (MSE) as a drift metric
    mse_val = mean_squared_error(y_val, y_pred_val)

    return mse_val

# Function to check for concept drift
def check_concept_drift(X_val, y_val, model, baseline_mse, drift_threshold):
    current_mse = monitor_drift(X_val, y_val, model)
    print("current mse:")
    print(current_mse)
    # Calculate the relative change in MSE
    relative_change = (current_mse - baseline_mse) / baseline_mse
    print("change:")
    print(relative_change)
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
    run_id = "d85bec5a35714e5ab3f13228e7fb9fe3"
    model_uri = f"runs:/{run_id}/best_model"

    # Load the best model from the MLflow Model Registry
    model = mlflow.sklearn.load_model(model_uri)

    # Monitor drift using a subset of the training data as the "current validation" set
    baseline_mse = monitor_drift(X_train, y_train, model)
    print("base mse:")
    print(baseline_mse)
    # Define drift monitoring parameters
    drift_threshold = 1

    # Check for concept drift
    if check_concept_drift(X_val, y_val, model, baseline_mse, drift_threshold):
        print("Concept drift detected. Retraining the model...")
        
        # Retrain the model
       # model = retrain_model(X_train, y_train, model)
        
        # Update baseline MSE after retraining
        #baseline_mse = monitor_drift(X_val, y_val, model)
        
        # Update last retraining timestamp
        #last_retraining_timestamp = datetime.now()

        #print("Model retrained successfully.")
    else:
        print("no drift")

