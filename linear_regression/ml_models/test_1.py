from linear_regressor import LinearRegressor
import numpy as np

X = np.array([[1, 4, 7, 10, 13],
             [2, 5, 8, 11, 14],
             [3, 6, 9, 12, 15]])

y = np.array([[6], [15], [24], [33], [42]])

reg = LinearRegressor(regularization='l1', stochastic=True)
reg.fit(X, y)

print(y - X.T @ reg.theta - reg.b)

