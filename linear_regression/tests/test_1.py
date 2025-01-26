import os
import sys
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(current_directory, '../ml_model')
directory = os.path.abspath(directory)

sys.path.append(directory)

from linear_regressor import LinearRegressor
import numpy as np

X = np.array([[1, 4, 7, 10, 13],
             [2, 5, 8, 11, 14],
             [3, 6, 9, 12, 15]])

y = np.array([[6], [15], [24], [33], [42]])

reg = LinearRegressor(regularization='l1')
reg.fit(X, y, verbose=False)
print(reg.Rsquared(X, y))
print(reg.MSE(X, y))


