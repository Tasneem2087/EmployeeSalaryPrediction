import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# Load full adult.csv dataset
data = pd.read_csv("adult.csv")
data = data.replace(' ?', pd.NA).dropna()
data = data.drop(columns=['education'])  # redundant

# Encode categorical columns
label_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
data['income'] = target_encoder.fit_transform(data['income'])

# Train-test split
X = data.drop(columns=['income'])
y = data['income']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = KNeighborsClassifier()
model.fit(xtrain, ytrain)

# Save model, encoders, scaler, target encoder
joblib.dump(model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("✅ Model, scaler, and encoders saved successfully.")
