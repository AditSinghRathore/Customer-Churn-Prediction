import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import preprocess_data


def train_model():

    # Get preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Create Logistic Regression model
    model = LogisticRegression(random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")

    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save trained model
    joblib.dump(model, "models/logistic_regression.pkl")

    print("Logistic Regression model saved successfully.")

    return model


if __name__ == "__main__":
    train_model()