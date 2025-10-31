import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print(sigmoid(x))