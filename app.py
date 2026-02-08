from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("my_trained_model.pkl")
scaler = joblib.load("scaler.pkl")
fuel_encoder = joblib.load("fuel_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        fuel_type = request.form["fuel"]
        fuel_used = float(request.form["fuel_used"])
        distance = float(request.form["distance"])
        avg_mileage = float(request.form["avg_mileage"])

        # Derived features
        current_mileage = distance / fuel_used
        mileage = current_mileage
        threshold = 0.8 * avg_mileage
        deviation = avg_mileage - current_mileage
        mileage_drop = 1 if current_mileage < threshold else 0

        fuel_encoded = fuel_encoder.transform([fuel_type])[0]

        # 11 FEATURES (MATCH TRAINING SHAPE)
        features = np.array([[
            fuel_used,
            distance,
            mileage,
            avg_mileage,
            deviation,
            current_mileage,
            threshold,
            mileage_drop,
            fuel_encoded,
            0,   # extra feature
            0    # extra feature
        ]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] * 100

        result = "Fuel Misuse Detected 🚨" if prediction == 1 else "Normal Vehicle Behavior ✅"
        color = "red" if prediction == 1 else "green"

        return render_template("index.html",
                               result=result,
                               prob=round(probability),
                               color=color,
                               cm=round(current_mileage,2),
                               dev=round(deviation,2),
                               th=round(threshold,2))
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
