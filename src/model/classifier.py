import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

def classify(X_train, y_train, X_test, y_test):
    # Create a Gaussian Classifier
    nbClassifier = GaussianNB()
    nbClassifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nbClassifier.predict(X_test)
    
    #lets see the actual and predicted value side by side
    y_compare = np.vstack((y_test,y_pred)).T
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    #finding accuracy from the confusion matrix.
    accuracy = accuracy_score(y_test, y_pred)

    return y_pred, y_compare, cm, accuracy