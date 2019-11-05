import numpy as np

def relu(x):
    alpha = 0.5
    return np.where(x<0, 0.5*x, x)    

a = np.array([-1, -2, 0, 1, 2], dtype=np.float)
# print(a[a>0])
print(relu(a))