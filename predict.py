import joblib
import pandas as pd


# Load saved files once when the application starts
model = joblib.load("models/logistic_regression.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")


def predict_churn(customer_data):
    """
    Predict whether a customer will churn.

    Parameters:
        customer_data (dict): Dictionary containing customer details.

    Returns:
        str: "Yes" if customer is likely to churn,
             "No" otherwise.
    """

    # Convert dictionary to DataFrame
    df = pd.DataFrame([customer_data])

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    # Encode categorical columns
    for column, encoder in encoders.items():

        if column in df.columns:

            df[column] = encoder.transform(df[column].astype(str))

    # Scale features
    scaled_data = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled_data)[0]

    # Convert prediction back to readable form
    if prediction == 1:
        return "Yes"

    return "No"


# Test the file directly
if __name__ == "__main__":

    sample_customer = {

        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.30,
        "TotalCharges": 783.60
    }

    result = predict_churn(sample_customer)

    print("Prediction:", result)