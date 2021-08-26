import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#By RIshar

data=pd.read_csv("d.csv").to_numpy()
print(data.shape)
#print(data)
Degree=4
n=Degree+1
L=0.003
epochs=10000

#This is the function that generated the data
def g(i):
	return 100*(i*0.1-0.3)*(i*0.1-0.6)*i*0.1

#Horner's Method for Evaluating Polynomials.
def f(inp,alpha):
	p=alpha[n-1]
	for i in range(n-1):
		p=alpha[(n-2)-i]+inp*p
	return p

#Returns the Mean Squared Error.
def MSE(d,alpha):
	e=f(d[:,0],alpha)-d[:,1]
	return np.dot(e,e)/n

#This is probably really slow. Oh well.
def gradient(d,alpha):
	g=np.array([])
	X=d[:,0]
	Y=d[:,1]
	for i in range(alpha.size):
		g=np.append(g,np.sum((f(X,alpha)-Y)*np.power(X,i))/n)
	return g

x=data[:,0]
a=np.zeros(n)
original_error=MSE(data,a)
a[0]=20
print("""Prediction Coefficients:""")
print(a)
print("""Mean Squared Error:""")
print(MSE(data,a))
print("""Gradient Vector:""")
print(gradient(data,a))

for i in range(epochs):
	a=a-gradient(data,a)*L
	#print(MSE(data,a))
	if i%(epochs/5)==-1:
		plt.plot(x,f(x,a))

print("""Final Coefficients:""")
print(a)
print("""Original Error MSE:""")
print(original_error)
plt.scatter(x,data[:,1])

plt.plot(x,g(x),'b+')
plt.plot(x,f(x,a),'r')
plt.show()