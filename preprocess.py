import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess_data():

    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # encoding
    encoder = LabelEncoder()

    categorical_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column].astype(str))

    print(df.head())
    print()
    print(df.dtypes)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()

    