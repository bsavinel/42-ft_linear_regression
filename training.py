print("training.py file loading...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

#!############################################################################################!#
#!########################################  Function  ########################################!#
#!############################################################################################!#

def ft_progress(lst):
    with tqdm(
        lst, 
        bar_format="ETA: {remaining_s:.2f}s [{percentage:3.0f}%][{bar}] {n_fmt}/{total_fmt} | elapsed time {elapsed_s:.2f}s"
    ) as progress:
        for i in progress:
            yield i

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

def isVector(x):
	if ((isinstance(x, np.ndarray)) and (x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1))):
		return True
	return False

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

def fit(x, y, theta, alpha, max_iter):
	if (not check_matrix(x, -1, 1) or not check_matrix(y, x.shape[0], 1) or not check_matrix(theta, 2, 1)):
		return None
	copyTheta = theta.copy()
	for i in ft_progress(range(max_iter)):
		copyTheta = copyTheta - (alpha * simple_gradient(x, y, copyTheta))
	return copyTheta

def plot(x, y, xNorm, yNorm, theta, title = "Linear Regression", xlabel = "x", ylabel = "y", legend_pred = "Predicted value", legend_data = "Data value"):
	if (not check_matrix(x, -1,1 ) or not check_matrix(y,x.shape[0], 1) or not check_matrix(theta, 2, 1)):
		return
	if ((not isinstance(title, str)) or (not isinstance(xlabel, str)) or (not isinstance(ylabel, str))):
		return
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(x.reshape((-1)), denormalizer(predict(xNorm, theta), y), 'sy--')
	plt.plot(x.reshape((-1)), y.reshape((-1)), 'bo')
	plt.grid()
	plt.legend([legend_pred, legend_data])
	plt.show()

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
	print("Theta after training on normalized dataset is :", theta)
	plot(x, y, xNorm, yNorm, theta, xlabel=data.columns.values[0], ylabel=data.columns.values[1])
	print(predict(xNorm, theta)[0])
	print(denormalizer(predict(xNorm, theta), y)[0])
