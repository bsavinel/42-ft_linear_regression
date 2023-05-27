print("precision.py file loading...")

import numpy as np
import pandas as pd
from math import sqrt
from tqdm import tqdm

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def check_matrix(m, sizeX, sizeY, dim = 2):
	if (not isinstance(m, np.ndarray)):
		return False
	if (m.ndim != dim or m.size == 0):
		return False
	if (sizeX != -1 and m.shape[0] != sizeX):
		return False
	if (dim == 2 and sizeY != -1 and m.shape[1] != sizeY):
		return False
	return True

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

def normalizer(x, list1):
	if (not isinstance(x, np.ndarray) or x.size == 0):
		return None
	if (x.ndim != 1 and not (x.ndim == 2 and x.shape[1] == 1)):
		return None
	Xcopy = x.reshape(-1)
	min = np.min(list1)
	max = np.max(list1)
	return (Xcopy - min) / (max - min)

def denormalizer(x, list1):
	if (not isinstance(x, np.ndarray) or x.size == 0):
		return None
	if (x.ndim != 1 and not (x.ndim == 2 and x.shape[1] == 1)):
		return None
	Xcopy = x.reshape(-1)
	min = np.min(list1)
	max = np.max(list1)
	return Xcopy * (max - min) + min

def simple_gradient(x, y, theta):
	if (not check_matrix(x, -1, 1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, 2, 1)):
		return None
	copyX = np.insert(x, 0, 1, axis=1)
	transpX = copyX.transpose()
	return (transpX @ (copyX @ theta - y)) / x.shape[0]

def ft_progress(lst):
    with tqdm(
        lst, 
        bar_format="ETA: {remaining_s:.2f}s [{percentage:3.0f}%][{bar}] {n_fmt}/{total_fmt} | elapsed time {elapsed_s:.2f}s"
    ) as progress:
        for i in progress:
            yield i

def fit(x, y, theta, alpha, max_iter):
	if (not check_matrix(x, -1, 1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, 2, 1)):
		return None
	copyTheta = theta.copy()
	for i in ft_progress(range(max_iter)):
		copyTheta = copyTheta - (alpha * simple_gradient(x, y, copyTheta))
	return copyTheta

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

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	for i in range(4):
		if (i == 3):
			print("To retry, relaunch the program")
			exit(1)
		try:
			file = str(input("\nFile name: "))
			if (isinstance(file, str) and file.endswith(".csv")):
				print("Try to read csv file...")
				data = pd.read_csv(file)
				print("csv file is good")
				break
			print("Invalid file name, please give a csv file")
		except:
			print("Fail to read csv file")
			pass

	for i in range(4):
		if (i == 3):
			print("To retry, relaunch the program")
			exit(1)
		try:
			learningRate = float(input("learningRate for normalized dataset: "))
			iteration = int(input("iteration: "))
			break
		except:
			print("Invalid value")

	theta0 = 0
	theta1 = 0
	x = np.array(data[data.columns.values[0]]).reshape(-1, 1)
	y = np.array(data[data.columns.values[1]]).reshape(-1, 1)
	xNorm = normalizer(x, x).reshape(-1, 1)
	yNorm = normalizer(y, y).reshape(-1, 1)
	theta = np.array([[0.],[0.]])
	theta = fit(xNorm, yNorm, theta, learningRate, iteration)
	prediction = predict(x, theta)
	prediction = denormalizer(prediction, y)
	y = y.reshape(-1)
	yNorm = yNorm.reshape(-1)
	predictionNorm = normalizer(prediction, y)
	print("\nThe mse of the model :", mse(y, prediction))
	print("The rmse of the model :", rmse(y, prediction))
	print("The mae of the model :", mae(y, prediction))
	print("The r2score of the model :", r2score(y, prediction))
	print("\nThe mse of the model normalized :", mse(yNorm, predictionNorm))
	print("The rmse of the model normalized :", rmse(yNorm, predictionNorm))
	print("The mae of the model normalized :", mae(yNorm, predictionNorm))
	print("The r2score of the model normalized :", r2score(yNorm, predictionNorm))