import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib


# Generate dummy data
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Save CSV for testing
df.drop(columns=['target']).to_csv("sample_data.csv", index=False)

# Train a simple model
model = RandomForestClassifier()
model.fit(df[feature_names], df['target'])

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model and sample_data.csv saved.")
