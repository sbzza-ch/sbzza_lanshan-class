"""mport numpy as np
import matplotlib.pyplot as plt
x=np.arange(-10,10,0.1)
#x=np.array(-5,5,dtype='float64')
y=np.where(x>0,1,0)
plt.plot(x,y)
plt.show()"""
import numpy as np
x=np.array([-1.0,1.0,2.0])
print(x)
y=x>0
print(y)
y=y.astype(int)
print(y)