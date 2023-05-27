import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from utils import check_matrix, ft_progress, updateTheta

#!############################################################################################!#
#!#######################################  Prediction  #######################################!#
#!############################################################################################!#

def add_intercept(x):
	if ((not isinstance(x, np.ndarray)) or x.size == 0):
		return None
	tmp = x.copy()
	if (tmp.ndim == 1):
		tmp.resize((tmp.shape[0], 1))
	return np.insert(tmp, 0, 1, axis=1)

def predict(x, theta):
	if (not check_matrix(x, -1, 1) or not check_matrix(theta, 2, 1)):
		return None
	return np.dot(add_intercept(x), theta)

def simple_gradient(x, y, theta):
	if (not check_matrix(x, -1, 1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, 2, 1)):
		return None
	copyX = np.insert(x, 0, 1, axis=1)
	transpX = copyX.transpose()
	return (transpX @ (copyX @ theta - y)) / x.shape[0]

def fit(x, y, theta, alpha, max_iter, x_norm=None, y_norm=None):
	if (not check_matrix(x, -1, 1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, 2, 1)):
		return None
	if (x_norm is not None):
		x_norm = x
	if (y_norm is not None):
		y_norm = y

	copyTheta = theta.copy()
	for i in ft_progress(range(max_iter)):
		copyTheta = copyTheta - (alpha * simple_gradient(x_norm, y_norm, copyTheta))
		updateTheta(copyTheta, x, y, predict(x, copyTheta))
	return copyTheta

#!#########################################################################################!#
#!######################################  Precision  ######################################!#
#!#########################################################################################!#

def mse(y, y_hat):
	if (not check_matrix(y, -1, -1, 1) or not check_matrix(y_hat, y.shape[0], -1, 1)):
		return None
	return sum((y_hat - y) ** 2) / y.size

def rmse(y, y_hat):
	ret = mse(y, y_hat)
	if (ret is None):
		return None
	return sqrt(ret)

def mae(y, y_hat):
	if (not check_matrix(y, -1, -1, 1) or not check_matrix(y_hat, y.shape[0], -1, 1)):
		return None
	return sum(abs(y_hat - y)) / y.size

def r2score(y, y_hat):
	if (not check_matrix(y, -1, -1, 1) or not check_matrix(y_hat, y.shape[0], -1, 1)):
		return None
	return 1 - (sum((y_hat - y) ** 2) / sum((y - y.mean()) ** 2))

#!#############################################################################################!#
#!######################################  Visualisation  ######################################!#
#!#############################################################################################!#

def plot(x, y, theta, title = "Linear Regression", xlabel = "x", ylabel = "y", legend_pred = "Predicted value", legend_data = "Data value"):
	if (not check_matrix(x, -1,1 ) or not check_matrix(y,x.shape[0], 1) or not check_matrix(theta, 2, 1)):
		return
	if ((not isinstance(title, str)) or (not isinstance(xlabel, str)) or (not isinstance(ylabel, str))):
		return
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(x.reshape((-1)), predict(x, theta), 'sy--')
	plt.plot(x.reshape((-1)), y.reshape((-1)), 'bo')
	plt.grid()
	plt.legend([legend_pred, legend_data])
	plt.show()
