import os
import joblib

from sklearn.tree import DecisionTreeClassifier

from preprocess import preprocess_data


def train_model():

    # Get processed data
    X_train, X_test, y_train, y_test, scaler = preprocess_data()

    # Create model
    model = DecisionTreeClassifier(random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save trained model
    joblib.dump(model, "models/decision_tree.pkl")

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    return model, X_test, y_test


if __name__ == "__main__":

    model, X_test, y_test = train_model()

    print(model)