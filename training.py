import numpy as np
import pandas as pd
from utils import  normalizer, denormalizer, deNormTheta, storeTheta
from linear_regraision import fit, plot, predict

for i in range(4):
	if (i == 3):
		print("To retry, relaunch the program")
		exit(1)
	try:
		file = str(input("\nFile name: "))
		if (isinstance(file, str) and file.endswith(".csv")):
			print("Try to read csv file...")
			data = pd.read_csv(file)
			x = np.array(data["km"]).reshape(-1, 1)
			y = np.array(data["price"]).reshape(-1, 1)
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
		value = int(input("Mileage for prediction: "))
		break
	except:
		print("Invalid value")

xNorm = normalizer(x, x).reshape(-1, 1)
yNorm = normalizer(y, y).reshape(-1, 1)
theta = np.zeros((2, 1))
theta = fit(xNorm, yNorm, theta, learningRate, iteration)
pred = denormalizer(predict(xNorm, theta), y).reshape(-1, 1)
newTheta = deNormTheta(x, pred)
storeTheta(newTheta)
prediction = newTheta[0][0] + (newTheta[1][0] * value)
print("The estimate price is : ", prediction)
plot(x, y, newTheta.reshape(-1,1), xlabel="mileage", ylabel="price")

