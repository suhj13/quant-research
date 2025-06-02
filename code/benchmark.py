import numpy as np

def sum(p):
    res = 0
    for i in p:
        res += i

    return res

n = 1000000
p = np.random.random(n)

print(sum(p))