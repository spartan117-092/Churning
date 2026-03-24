from flask import Flask, render_template, request
from preprocessing import preprocess
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("pipeline.pkl")
preprocess = pipeline["preprocessor"]
models = pipeline["models"]

app = Flask(__name__)

# Home page with form
@app.route("/", methods=["GET", "POST"])
def home():
    prediction_result = None

    if request.method == "POST":
        # Collect form data
        data = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": int(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        # Convert to DataFrame and preprocess
        df = pd.DataFrame([data])
        df = preprocess(df)
        if "Churn" in df.columns:
            df = df.drop("Churn", axis=1)

        # Make predictions
        prediction_result = {}
        for name, model in models.items():
            pred = int(model.predict(df)[0])
            prob = float(model.predict_proba(df)[0][1])
            prediction_result[name] = {"prediction": pred, "probability": prob}

    return render_template("index.html", result=prediction_result)


if __name__ == "__main__":
    app.run(debug=True)