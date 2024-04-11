from src.model.classifier import classify
from src.preprocessing.processor import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = 'data\\iris.csv'
data = pd.read_csv(path)

# ____ Visualize data ____

# Bar chart 
sns.countplot(data=data, x='species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.title('Number of Each Species')
plt.show()

# Scatter plot 
plt.scatter(data['sepal_length'], data['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Comparison of Sepal Length and Sepal Width')
plt.show()

# Scatter plot 2
plt.scatter(data['petal_length'], data['petal_width'])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Comparison of Petal Length and Petal Width')
plt.show()

# Box plot
sns.boxplot(data=data, x='species', y='petal_length')
plt.xlabel('Species')
plt.ylabel('Petal Length')
plt.title('Distribution of Petal Length by Species')
plt.show()

# Heatmap 
numeric_attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
correlation_matrix = data[numeric_attributes].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# ____ Main ____
    
X_train, X_test, y_train, y_test = preprocessing(path)
y_pred, y_compare, cm, accuracy = classify(X_train, y_train, X_test, y_test)

print('Predicted labels:\n', y_pred)
print('\nActual labels:\n', y_compare)
print('\nConfusion Matrix:\n', cm)
print('\nAccuracy:', accuracy)