# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the raw sensor data
raw_data = pd.read_csv('dummy_sensor_data.csv')

# Assuming 'Reading' is the target variable, separate features and target
X = raw_data.drop('Reading', axis=1)
y = raw_data['Reading']

# Normalize or scale features (you can choose the appropriate scaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save the preprocessed data
pd.DataFrame(X_train).to_csv('preprocessed_X_train.csv', index=False)
pd.DataFrame(X_val).to_csv('preprocessed_X_val.csv', index=False)
pd.DataFrame(y_train).to_csv('preprocessed_y_train.csv', index=False)
pd.DataFrame(y_val).to_csv('preprocessed_y_val.csv', index=False)
