from src.model.classifier import classify
from src.preprocessing.processor import preprocessing

path = 'data\\iris.csv'

X_train, X_test, y_train, y_test = preprocessing(path)
y_pred, y_compare, cm, accuracy = classify(X_train, y_train, X_test, y_test)

print('Predicted labels:\n', y_pred)
print('\nActual labels:\n', y_compare)
print('\nConfusion Matrix:\n', cm)
print('\nAccuracy:', accuracy)