import numpy as np

a = [1,2,3]
print(a)

a = np.float16(a)
print(a)

b = np.float16(0)
print(b.dtype)

array1 = [0]*5
array2 = [np.float16(0)]*5

array1 = np.float16(array1)

print(array1.dtype)

array3 = np.copy(array1)
print(array3.dtype)

print(abs(-5))