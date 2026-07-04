## Live Link:
https://customer-churn-prediction-0vo0.onrender.com

## Introduction:
    Customer Churn Prediction is machine Learning model using Logistic regression model to predict Customer Churn Risk. Project is tested and deployed with the best performing model with accuracy of 81.54 %. 

## Tech Stack:

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- Git
- GitHub
- Render

## Architechture:
    3 layer architechture:
        
        1.Presentation layer: Frontend
            - HTML form collects input for prediction send it to flask
        2.Application layer: Flask
            - Recieves request.
            - Validates input.
            - calls the prediction module.
        3.Machine Learning layer
            - Loads the trained Logistic Regression model.
            - Predicts customer churn.
            - Returns the prediction.

## WorkFlow:

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

## Project Structure:
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