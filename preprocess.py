import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('./data/product_sample.csv')

# Drop missing values
df = df.dropna()
df = df.drop('weekday', axis=1)
df['date'] = pd.to_datetime(df['selling_date'])


# Standardize numerical features
scaler = StandardScaler()
numerical_features = ['total_salse', 'product_id']  # Replace with your actual numerical features
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Save the preprocessed data
df.to_csv('data/processed_data.csv', index=False)
