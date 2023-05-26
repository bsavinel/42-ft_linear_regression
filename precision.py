print("precision.py file loading...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

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
			isNormalised = input("\nDo you want use on normalize dataset ?(Y/N) ").capitalize()
			if (isNormalised != "Y" and isNormalised != "N"):
				raise ValueError
			break
		except:
			print("Invalid value")

	for i in range(4):
		if (i == 3):
			print("To retry, relaunch the program")
			exit(1)
		try:
			theta0 = float(input("\ntheta0 for normalized dataset: "))
			theta1 = float(input("theta1 for normalized dataset: "))
		except:
			print("Invalid value")
	
	x = np.array(data[data.columns.values[0]]).reshape(-1, 1)
	y = np.array(data[data.columns.values[1]])
	xUse = x
	yUse = y
	if (isNormalised == "Y"):
		xUse = normalizer(x, x)
		yUse = normalizer(y, y)
	theta = np.array([[theta0],[theta1]])
	prediction =  predict(x, theta)
	if (isNormalised == "Y"):
		prediction = denormalizer(prediction, y)
	print("\nThe mse of the model :", mse(y, prediction))
	print("\nThe rmse of the model :", rmse(y, prediction))
	print("\nThe mae of the model :", mae(y, prediction))
	print("\nThe r2score of the model :", r2score(y, prediction))