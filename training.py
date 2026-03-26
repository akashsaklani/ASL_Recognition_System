import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset.csv", header=None)

# Split features and labels
X = data.iloc[:, :-1]   # first 42 columns
y = data.iloc[:, -1]    # last column (label)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")