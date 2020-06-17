import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv('student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
# print(data)

predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

max_accuracy = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)
    print('Accuracy:', accuracy)

    # print('Coef:', linear.coef_)
    # print('Intercept', linear.intercept_)

    if accuracy > max_accuracy:
        max_accuracy = accuracy

        # Saving Best Models
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)

pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

# Reopen Our saved Models
print('Coef:', linear.coef_)
print('Intercept', linear.intercept_)

predictions = linear.predict(x_test)
print(predictions)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Drawing and plotting model
style.use('ggplot')
p = 'G1'
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()

