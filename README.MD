

    Architechture:
    3 layer architechture:
        
        1.Presentation layer: Frontend
            - HTML form accepts the prediction details
        2.Application layer: Flask
            - Recieves request.
            - Validates input.
            - calls the prediction module.
        3.Machine Learning layer
            - Loads the trained Logistic Regression model.
            - Predicts customer churn.
            - Returns the prediction.

    User
    │
    ▼
    HTML Form
    │
    ▼
    Flask (app.py)
    │
    ▼
    predict.py
    │
    ▼
    Logistic Regression Model (.pkl)
    │
    ▼
    Prediction

    Project Structure:
    CustomerChurnPrediction/
    │
    ├── app.py
    ├── predict.py 
    ├── preprocess.py : preprocesses the data
    ├── train.py : Trains the machine learning model and saves the selected model as a .pkl file.(Model Accuracy: 0.8155)
    ├── comparison.py : used for comparing models
    │
    ├── models/
    │   ├── logistic_regression.pkl : Primary model
    │   ├── decision_tree.pkl : Primary model
    │   └── scaler.pkl : saves the scalar parameters for the model after preprocessing
    │
    ├── data/
    │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
    │
    ├── requirements.txt
    └── README.md