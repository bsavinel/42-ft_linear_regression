from utils import loadTheta

for i in range(4):
	if (i == 3):
		print("To retry, relaunch the program")
		exit(1)
	try:
		value = int(input("mileage: "))
		break
	except:
		print("Invalid value")
theta = loadTheta()
print("The estimate price is : ", theta[0][0] + theta[1][0] * value)
