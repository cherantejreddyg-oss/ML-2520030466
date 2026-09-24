import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[1], [2], [3],[4],[5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(x, y)

x_test = np.array([[6]])
y_pred = model.predict(x_test)

print(y_pred)
