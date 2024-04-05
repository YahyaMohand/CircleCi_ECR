import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the preprocessed data
df = pd.read_csv('data/processed_data.csv')

# Split the data into features and target
X = df.drop('total_salse', axis=1)
y = df['total_salse']

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, './models/price_predictor.joblib')
