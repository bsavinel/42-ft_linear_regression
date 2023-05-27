from utils import loadTheta

print("This programe will give the price of a car in function of its mileage")

for i in range(4):
	if (i == 3):
		print("To retry, relaunch the program")
		exit(1)
	try:
		value = int(input("\nMileage : "))
		break
	except:
		print("Invalid value")
theta = loadTheta()
print("The estimate price is : ", theta[0][0] + theta[1][0] * value)
