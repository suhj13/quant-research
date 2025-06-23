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
print(array2[0].dtype)

array3 = np.copy(array1)
print(array3.dtype)

print(np.float16(min(4,6)))

sum = 0
for i in range(6):
    sum+= (i+0.422)
print(sum)

eps = np.finfo(np.float16).eps/2                    # The unit roundoff when using float16 values
eta = np.nextafter(np.float16(0), np.float16(1))    # Smallest possible number
print(eps, eta)
