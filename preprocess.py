import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data():

    # Load dataset
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Remove customer ID
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Dictionary to store encoders
    encoders = {}

    # Encode categorical columns
    categorical_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column].astype(str))

        encoders[column] = encoder

    # Features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Save scaler and encoders
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_data()