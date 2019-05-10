import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):

    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    # Adding the second hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    # Adding the output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier


classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 700],
              'optimizer': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',
                                cv=10)

grid_search.fit(X_train,y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
y_pred = grid_search.predict(X_test)
y_pred = (y_pred > 0.5)

