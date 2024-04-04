import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessing(data_path):
    # Read the csv file
    data = pd.read_csv(data_path)
    
    # As the given dataset is already clean, we don't need to clean it.

    # Splitting the dataset in independent and dependent variables
    X = data.iloc[:,:4].values
    y = data['species'].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)
    
    # Feature Scaling to bring the variable in a single scale
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test

# dhktqd.cntt.udttnt