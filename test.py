import numpy as np
x = np.array([1,2,3,4,5])
y = np.copy(x)

x[1] = 6

print(x)
print(y)

print(len(x))