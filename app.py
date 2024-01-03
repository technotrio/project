from flask import Flask, render_template
import mlflow.sklearn
import pandas as pd
import mlflow
import os
from preprocess import preprocess_data

app = Flask(__name__)

# Specify the MLflow experiment name and run ID
mlflow_experiment_name = "bittersweet-deer-731"
run_id = "0083f08e55b3466e9bf73a41b22e01f1"
artifact_uri = f"file:///mlflow/mlruns/0/{run_id}/artifacts/best_model"
loaded_model = mlflow.sklearn.load_model(artifact_uri)

def predict():
    # Get the input data
    temp_csv_path = 'input_data.csv'
    X, _ = preprocess_data(temp_csv_path)
    predictions = loaded_model.predict(X)

    # Log the predictions as a metric
    mlflow.log_metric("prediction_mean", predictions.mean())

    return predictions.tolist()

@app.route('/')
def index():
    predictions = predict()
    return render_template('index.html', predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True, port=8080, host='0.0.0.0')
