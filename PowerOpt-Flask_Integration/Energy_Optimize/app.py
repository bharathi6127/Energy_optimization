from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils.preprocess import preprocess_data, extract_date_features

app = Flask(__name__)

MODEL_PATH = "model/trained_model.pkl"
ENCODER_PATH = "model/encoder.pkl"
OUTLIER_PATH = "model/outlier.pkl"
SCALER_PATH = "model/scaler.pkl"

DATA_PATH = "data/dataset.csv"

# Train Route
@app.route("/train", methods=["POST"])
def train():
    try:
        # Load dataset
        df = pd.read_csv(DATA_PATH)

        # Preprocess data
        X, y = preprocess_data(df)

        # Train Model
        #Splitting data set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

        # Initialize and fit the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate R²
        r2 = r2_score(y_test, y_pred)

        # Calculate Adjusted R²
        n = X_test.shape[0]  # Number of samples
        p = X_test.shape[1]  # Number of features
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        # Print the results
        print(f"R² Score for Linear Regression: {r2:.4f}")
        print(f"Adjusted R² Score for Linear Regression: {adjusted_r2:.4f}")

        # Save trained model
        joblib.dump(model, MODEL_PATH)

        return jsonify({"message": f"R² Score for Linear Regression: {r2:.4f} Adjusted R² Score for Linear Regression: {adjusted_r2:.4f} Model trained and saved successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)})

# Predict Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json
        df = pd.DataFrame([input_data])

        # Convert date
        df = extract_date_features(df)

        # Load preprocessing models
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Apply preprocessing
        df = preprocess_data(df, training=False)

        # Load trained model
        model = joblib.load(MODEL_PATH)

        # Predict
        prediction = model.predict(df)

        return jsonify({"Usage_kWh": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)