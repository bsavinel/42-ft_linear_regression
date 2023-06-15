import numpy as np
import pandas as pd
from utils import loadTheta
from linear_regraision import predict, mae, mse, rmse, r2score, data_spliter

print("This programe will give the precision(mse, rmse, mae and r2score) of the model in function of a csv file")

for i in range(4):
	if (i == 3):
		print("To retry, relaunch the program")
		exit(1)
	try:
		file = str(input("\nFile name : "))
		if (isinstance(file, str) and file.endswith(".csv")):
			print("Try to read csv file...")
			seed = int(input("Seed : "))
			data = pd.read_csv(file)
			x = np.array(data["km"]).reshape(-1, 1)
			y = np.array(data["price"]).reshape(-1, 1)
			if (len(x) == 1):
				print("Dataset size is too small, please give a bigger")
				continue
			print("csv file is good")
			break
		print("Invalid file name, please give a csv file")
	except:
		print("Fail to read csv file")
		pass

xTrain, xEval, yTrain, yEval = data_spliter(x, y, 0.8, seed)

theta = loadTheta()
prediction = predict(xEval, theta).reshape(-1)
yEval = yEval.reshape(-1)

print("\nThe mse of the model :", mse(yEval, prediction))
print("The rmse of the model :", rmse(yEval, prediction))
print("The mae of the model :", mae(yEval, prediction))
print("The r2score of the model :", r2score(yEval, prediction))