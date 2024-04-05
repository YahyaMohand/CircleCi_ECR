import pandas as pd
import joblib

# Load the trained model
model = joblib.load('models/price_predictor.joblib')

# Load new data (for demonstration, we'll use the same preprocessed data)
new_data = pd.read_csv('data/processed_data.csv')
X_new = new_data.drop('total_salse', axis=1)

# Make predictions
predictions = model.predict(X_new)

# Save or display the predictions
print(predictions)
