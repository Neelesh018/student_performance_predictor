from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

FEATURES = [
    "Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly",
    "Absences", "Tutoring", "ParentalSupport", "Extracurricular",
    "Sports", "Music", "Volunteering"
]

NUMERIC_FIELDS = ["Age", "StudyTimeWeekly", "Absences"]

# Manual encodings (must match your training data)
manual_encoders = {
    "Gender": {"Male": 1, "Female": 0},
    "Tutoring": {"Yes": 1, "No": 0},
    "ParentalSupport": {"Yes": 1, "No": 0},
    "Extracurricular": {"Yes": 1, "No": 0},
    "Sports": {"Yes": 1, "No": 0},
    "Music": {"Yes": 1, "No": 0},
    "Volunteering": {"Yes": 1, "No": 0},
    # Add other mappings if needed
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            input_data = []
            for field in FEATURES:
                value = request.form.get(field.lower())

                if field in NUMERIC_FIELDS:
                    value = int(value)
                elif field in manual_encoders:
                    value = manual_encoders[field].get(value, 0)
                else:
                    # For Ethnicity or ParentalEducation — fallback to hash or label encoding later
                    value = hash(value) % 100  # quick fix (ensure consistent inputs)
                
                input_data.append(value)

            prediction = model.predict([input_data])[0]
            prediction = round(float(prediction), 2)

        except Exception as e:
            prediction = f"❌ Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
