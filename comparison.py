from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocess import preprocess_data

import joblib

print("======================================================")


#training logistic regression model

#get preprocessed data
X_train, X_test, y_train, y_test, scaler = preprocess_data()

#create model
lrmodel = LogisticRegression()

lrmodel.fit(X_train,y_train)

l_pred = lrmodel.predict(X_test)

lr_accuracy = accuracy_score(y_test,l_pred)
lr_precision = precision_score(y_test,l_pred)
lr_recall = recall_score(y_test,l_pred)
lr_f1score = f1_score(y_test,l_pred)
lr_ras = roc_auc_score(y_test,l_pred)

print("LR score:")

print("accuracy",lr_accuracy)
print("precision",lr_precision)
print("recall",lr_recall)
print("f1_score",lr_f1score)
print("roc_auc_score",lr_ras)

print("======================================================")

print("KNN score:")
#KNN

#create model

knn = KNeighborsClassifier(n_neighbors=5)

#train model

knn.fit(X_train,y_train)

k_pred = knn.predict(X_test)

k_accuracy = accuracy_score(y_test,k_pred)
k_precision = precision_score(y_test,k_pred)
k_recall = recall_score(y_test,k_pred)
k_f1score = f1_score(y_test,k_pred)
k_ras = roc_auc_score(y_test,k_pred)

print("accuracy",k_accuracy)
print("precision",k_precision)
print("recall", k_recall)
print("f1_score",k_f1score)
print("roc_auc_score",k_ras)

print("======================================================")

print("Decision tree score:")
model = joblib.load("models/decision_tree.pkl")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1score = f1_score(y_test,y_pred)
ras = roc_auc_score(y_test,y_pred)

print("accuracy",accuracy)
print("precision",precision)
print("recall", recall)
print("f1_score",f1score)
print("roc_auc_score",ras)

print("======================================================")