import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
import pandas
from sklearn.metrics import accuracy_score

data_train = pandas.read_csv(r'C:\Users\Dmitriy\PycharmProjects\perceptron-train.csv', header = None)
y_train = data_train[0]
x_train = data_train.loc[:, 1:]

data_test = pandas.read_csv(r'C:\Users\Dmitriy\PycharmProjects\perceptron-test.csv', header = None)
y_test = data_test[0]
x_test = data_test.loc[:, 1:]

clf = Perceptron(random_state=241, max_iter=5)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
print(accuracy)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

clf.fit(x_train_scaled, y_train)
predictions = clf.predict(x_test_scaled)

accuracy_after = accuracy_score(y_test, predictions)
print(accuracy_after)

print(accuracy_after-accuracy)

