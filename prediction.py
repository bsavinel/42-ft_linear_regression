print("prediction.py file loading...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

POSIBILITY_NORMALIZED = False

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

def denormalizer(x, list1):
	if (not isinstance(x, np.ndarray) or x.size == 0):
		return None
	if (x.ndim != 1 and not (x.ndim == 2 and x.shape[1] == 1)):
		return None
	Xcopy = x.reshape(-1)
	min = np.min(list1)
	max = np.max(list1)
	return Xcopy * (max - min) + min

def normalizer(x, list1):
	if (not isinstance(x, np.ndarray) or x.size == 0):
		return None
	if (x.ndim != 1 and not (x.ndim == 2 and x.shape[1] == 1)):
		return None
	Xcopy = x.reshape(-1)
	min = np.min(list1)
	max = np.max(list1)
	return (Xcopy - min) / (max - min)

#!####################################################################################################!#
#!##############################################  TEST  ##############################################!#
#!####################################################################################################!#

if __name__ == '__main__':
	if (not POSIBILITY_NORMALIZED):
		for i in range(4):
			if (i == 3):
				print("To retry, relaunch the program")
				exit(1)
			try:
				theta0 = float(input("\ntheta0: "))
				theta1 = float(input("theta1: "))
				value = int(input("value: "))
				break
			except:
				print("Invalid value")

		print("Result of the prediction : ", theta0 + theta1 * value)

	else :
		for i in range(4):
			if (i == 3):
				print("To retry, relaunch the program")
				exit(1)
			try:
				theta0 = float(input("\ntheta0: "))
				theta1 = float(input("theta1: "))
				value = int(input("value: "))
				break
			except:
				print("Invalid value")

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

		if (isNormalised == "Y"):
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
					print("Invalid value")

		if (isNormalised == "N"):
			print("Result of the prediction : ", theta0 + theta1 * value)
		else:
			y = np.array(data[data.columns.values[1]]).reshape(-1, 1)
			x = np.array(data[data.columns.values[0]]).reshape(-1, 1)
			value = normalizer(np.array([value]), x)
			pred = np.array([[theta0 + theta1 * value[0]]])
			print(pred[0])
			print("Result of the prediction : ", denormalizer(pred, y)[0])
