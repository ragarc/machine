import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = (12.0, 9.0)


def noise_shape(v,p,a):
	return ((np.power(v,p)+a*v)/(1+a))*noise

def g_f(i):
	return 100*(i-0.3)*(i-0.6)*i

n=1000
noise=20

x=np.sort(np.random.rand(n))
y=g_f(x)+noise_shape((np.random.rand(n)-0.5)*2,5,0.33)
print(np.array([x,y]))
#np.savetxt('d.csv',np.transpose(np.array([x,y])),delimiter=',')
plt.scatter(x,y)
plt.plot(x,g_f(x),'r')
plt.show()

