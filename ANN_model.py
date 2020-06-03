import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('credit_data_anomaly_detection_without_normalization.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'tanh', input_dim = 4))
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'tanh'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 500, nb_epoch = 150)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




















