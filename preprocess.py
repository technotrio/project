# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def preprocess_data(file_path):
    # Load the raw sensor data
    raw_data = pd.read_csv(file_path, parse_dates=['Timestamp'])

    # Separate features and target
    X = raw_data[['Timestamp', 'Machine_ID', 'Sensor_ID']]
    y = raw_data['Reading']

    # Function to extract time-related features from datetime
    def add_time_features(df):
        df['Hour'] = df['Timestamp'].dt.hour
        df['Minute'] = df['Timestamp'].dt.minute
        df['Second'] = df['Timestamp'].dt.second
        df['DayofWeek'] = df['Timestamp'].dt.dayofweek
        df['DayofMonth'] = df['Timestamp'].dt.day
        df['Month'] = df['Timestamp'].dt.month
        df['Year'] = df['Timestamp'].dt.year
        return df.drop('Timestamp', axis=1)

    # Define column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('timestamp', FunctionTransformer(add_time_features), ['Timestamp']),
            ('categorical', OneHotEncoder(), ['Machine_ID', 'Sensor_ID'])
        ],
        remainder='passthrough'
    )

    # Define the pipeline with preprocessing and scaling
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])

    # Preprocess and scale the features
    X_scaled = pipeline.fit_transform(X)

    return X_scaled, y
