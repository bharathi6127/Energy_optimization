import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Extract Date Features
def extract_date_features(df):
    if "date" not in df.columns:
        raise ValueError("Missing required column: 'date'")

    # Convert date column to datetime format
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # Check if conversion was successful
    if df["date"].isna().sum() > 0:
        raise ValueError("Some 'date' values could not be parsed. Check format.")

    # Extract features
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df["Day"] = df["date"].dt.day
    df["Hour"] = df["date"].dt.hour
    df["Minute"] = df["date"].dt.minute

    return df 

# Handle Categorical Encoding
def encode_features(df, training=True):
    categorical_cols = ['WeekStatus', 'Day_of_week', 'Load_Type']

    if training:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = encoder.fit_transform(df[categorical_cols])
        joblib.dump(encoder, "model/encoder.pkl")
    else:
        encoder = joblib.load("model/encoder.pkl")
        encoded = encoder.transform(df[categorical_cols])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df

def standardize_column_names(df):
    column_map = {
        "Lagging_Current_Reactive.Power_kVarh": "Lagging_Current_Reactive_Power_kVarh",
        "CO2(tCO2)": "CO2",
        "Date_Time": "date",  # If date column is named differently, rename it
        "Datetime": "date"
    }
    
    df.rename(columns=column_map, inplace=True)

    if "date" not in df.columns:
        raise ValueError("Missing required column: 'date' after renaming. Check dataset.")
    
    return df

# Handle Outliers
def handle_outliers(X, y=None, training=True):
    if training:
        iso = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso.fit_predict(X)
        joblib.dump(iso, "model/outlier.pkl")
    else:
        iso = joblib.load("model/outlier.pkl")
        outliers = iso.predict(X)

    mask = outliers == 1  # Keep only non-outliers
    X_cleaned = X[mask]

    if y is not None:
        y_cleaned = y[mask]
        return X_cleaned, y_cleaned
    return X_cleaned

# Scaling
def scale_features(X, training=True):
    if training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, "model/scaler.pkl")
    else:
        scaler = joblib.load("model/scaler.pkl")
        X_scaled = scaler.transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns)

# Main Preprocessing Function
def preprocess_data(df, training=True):
    df = standardize_column_names(df)  
    df = extract_date_features(df)  

    # Drop 'date' column for both training and prediction
    if "date" in df.columns:
        df.drop(columns=["date"], inplace=True)  

    y = df["Usage_kWh"] if "Usage_kWh" in df.columns else None
    X = df.drop(columns=["Usage_kWh"]) if "Usage_kWh" in df.columns else df

    X = encode_features(X, training)
    
    if training:
        X, y = handle_outliers(X, y, training)
    else:
        X = handle_outliers(X, training=training)

    X = scale_features(X, training)

    return (X, y) if training else X