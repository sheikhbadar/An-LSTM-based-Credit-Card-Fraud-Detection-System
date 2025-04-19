import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Create a simple model for testing
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create some sample data for training
X = np.random.randn(1000, 6)  # 6 features
y = np.random.randint(0, 2, 1000)  # Binary labels

# Fit the model
model.fit(X, y)

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(X)

# Save the model and scaler
pd.to_pickle(model, 'model.pkl')
pd.to_pickle(scaler, 'scaler.pkl')

print("Model and scaler created successfully!") 