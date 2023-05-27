import numpy as np
import pandas as pd
from utils import loadTheta
from linear_regraision import predict, mae, mse, rmse, r2score

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

x = np.array(data[data.columns.values[0]]).reshape(-1, 1)
y = np.array(data[data.columns.values[1]]).reshape(-1, 1).reshape(-1)
theta = loadTheta()
prediction = predict(x, theta).reshape(-1)

print("\nThe mse of the model :", mse(y, prediction))
print("The rmse of the model :", rmse(y, prediction))
print("The mae of the model :", mae(y, prediction))
print("The r2score of the model :", r2score(y, prediction))