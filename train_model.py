import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load and clean dataset
df = pd.read_csv("Student_performance_data _.csv")
df.columns = df.columns.str.strip()  # Remove any whitespace in column names

# Drop unused columns
df = df.drop(columns=["StudentID", "GradeClass"])

# Encode categorical features
categorical_columns = [
    "Gender", "Ethnicity", "ParentalEducation", "Tutoring",
    "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering"
]

encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Separate features and target
X = df.drop("GPA", axis=1)
y = df["GPA"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Custom regression accuracy (within ±0.5 GPA)
tolerance = 0.5
accuracy = np.mean(np.abs(y_pred - y_test) <= tolerance)

print(f"✅ R² Score: {r2:.4f}")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ Custom Accuracy (within ±{tolerance} GPA): {accuracy * 100:.2f}%")

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("✅ Model and encoders saved as model.pkl and encoders.pkl.")
