import numpy as np
from tqdm import tqdm
import pandas as pd

#!###########################################################################################!#
#!########################################  Utils  ########################################!#
#!###########################################################################################!#

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

def ft_progress(lst):
    with tqdm(
        lst, 
        bar_format="ETA: {remaining_s:.2f}s [{percentage:3.0f}%][{bar}] {n_fmt}/{total_fmt} | elapsed time {elapsed_s:.2f}s"
    ) as progress:
        for i in progress:
            yield i

#!##########################################################################################!#
#!######################################  Converteur  ######################################!#
#!##########################################################################################!#

def normalizer(x, list1):
	if (not isinstance(x, np.ndarray) or x.size == 0):
		return None
	if (x.ndim != 1 and not (x.ndim == 2 and x.shape[1] == 1)):
		return None
	Xcopy = x.reshape(-1)
	min = np.min(list1)
	max = np.max(list1)
	if (max == min):
		raise ValueError("Normalizer: max and min of the array are equal")
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

def deNormTheta(theta, valx, valy):
	if (not check_matrix(theta, 2, 1) or not check_matrix(valx, -1, 1) or not check_matrix(valy, -1, 1)):
		return None
	ret = np.zeros((2, 1))
	ret[1] = (valy[0] - valy[1]) / (valx[0] - valx[1])
	ret[0] = valy[0] - (ret[1] * valx[0])
	return ret

#!#########################################################################################!#
#!####################################  Theta Manager  ####################################!#
#!#########################################################################################!#

def storeTheta(theta):
	if (not check_matrix(theta, 2, 1)):
		return None
	theta = theta.reshape(-1)
	theta = theta.tolist()
	file = pd.DataFrame(theta)
	file.to_csv("theta.csv", index=False, header=False)
	return theta

def loadTheta():
	try:
		theta = pd.read_csv("theta.csv", header=None)
		theta = theta.to_numpy()
		return theta.reshape(-1, 1)
	except:
		storeTheta(np.zeros((2, 1)))
		return np.zeros((2, 1))
	
def updateTheta(theta, x, y, pred):
	predUnnorm = denormalizer(pred, y)
	storeTheta = deNormTheta(theta, x, predUnnorm)
	storeTheta = storeTheta(storeTheta)