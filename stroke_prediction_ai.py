import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load medical stroke dataset
df = pd.read_csv('stroke_data.csv')

# Feature Selection
X = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]
y = df['stroke']

# Split Data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize Machine Learning Model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict risk on test set
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("=== AI Model Performance ===")
print(f"Stroke Risk Prediction Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test AI on a new hypothetical patient
# [Age: 72, Hypertension: 1, Heart Disease: 0, Avg Glucose: 210, BMI: 33]
new_patient = [[72, 1, 0, 210, 33]]
prediction = model.predict(new_patient)
risk_status = "High Risk of Stroke" if prediction[0] == 1 else "Low Risk"
print(f"\nAI Diagnosis for New Patient: {risk_status}")