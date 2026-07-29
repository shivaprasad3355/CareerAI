import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("career_guidance_dataset_500.csv")

print("Dataset Loaded Successfully")

os.makedirs("models", exist_ok=True)

# Convert Yes/No columns into numbers
binary_columns = [
    "Python",
    "Java",
    "SQL",
    "C++",
    "HTML_CSS",
    "Internship",
    "Certifications"
]

for col in binary_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# Encode remaining text columns
encoders = {}

categorical_columns = [
    "Communication",
    "Aptitude",
    "Interest",
    "Career"
]

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# Features and target
X = df.drop("Career", axis=1)
y = df["Career"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("-------------------------")
print("Model Accuracy:", accuracy * 100, "%")
print("-------------------------")

# Save files
joblib.dump(model, "models/career_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("Model and encoders saved successfully!")