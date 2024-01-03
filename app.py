from flask import Flask, render_template
import mlflow.sklearn
import pandas as pd
import mlflow
import os
from preprocess import preprocess_data

app = Flask(__name__)
mlruns_path = os.path.join(os.getcwd(), 'mlruns')

dvc_command = "dvc repro"

latest_best_model_path = None
latest_run_id = None
for root, dirs, files in os.walk(mlruns_path):
    for dir_name in dirs:
        run_path = os.path.join(root, dir_name)
        artifact_path = os.path.join(run_path, 'artifacts', 'best_model')

        if os.path.exists(artifact_path):
            if latest_run_id is None or dir_name > latest_run_id:
                latest_run_id = dir_name
                latest_best_model_path = artifact_path

if latest_best_model_path:
    artifact_uri = f"file://{latest_best_model_path}"
    loaded_model = mlflow.sklearn.load_model(artifact_uri)
    print(latest_best_model_path)
else:
    print("No 'best_model' artifact found in mlruns.")

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
