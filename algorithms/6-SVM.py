import pandas
from sklearn.svm import SVC
import numpy as np

data = pandas.read_csv(r'C:\Users\Dmitriy\PycharmProjects\container.csv', header = None)
x_train = data.loc[:, 1:]
y_train = data[0]
trainer = SVC(kernel='linear', C = 1000)
trainer.fit(x_train, y_train)

print('test sample accuracy' + str(trainer.score()))
