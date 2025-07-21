import numpy as np
import pandas as pd
from math import fsum

# Utillize float16 as inputs
def fl(x):
    y = np.float16(x)
    return y

# Fasttwosum algorithm given |a| >= |b|
def fasttwosum(a, b):
    a = fl(a)
    b = fl(b)
    s = a + b   # sum in float16
    t = s - a   # lost portion of b due to rounding 
    error = b - t   # error term (a + b -s)
    return s, error


def FastPrecSum(p, K):
    # p is an np array of float16 values
    # K is an integer representing the no. of fixed extractions
    # Operations between two float16 values will return a float16 result
    n = len(p)
    ExactFlag = False

    eps = np.finfo(np.float16).eps/2                    # The unit roundoff when using float16 values
    eta = np.nextafter(np.float16(0), np.float16(1))    # Smallest possible number

    if K == 1:
        return(np.sum(p, dtype = np.float16))                       # Return the regular sum value 

    T_values = [fl(0)]*K
    T_values[0] = fl(np.sum(np.abs(p))/(1 - n*eps))
    
    sigma_0 = [fl(0)]*(K-1)
    sigma_tilde = [fl(0)]*(K-1)
    phi = [fl(0)]*(K-1)

    for i in range(K-1):
        sigma_0[i] = (2*T_values[i])/(1 - (3*n + 1)*eps)
        sigma_tilde[i] = sigma_0[i]

        q = (sigma_0[i] / (2*eps))
        u = np.abs(q/(1 - eps) - q)   # u = ufp(sigma_0)

        phi[i] = ((2*n*(n + 2)*eps)*u)/(1 - 5*eps)

        T_values[i+1] = min(((3/2 + 4*eps)*(n*eps)*sigma_0[i]),((2*n*eps)*u))   # Need confirmation on the calculation of this value

        if (4*T_values[i+1] <= (eta/eps)):
            K = i +1
            ExactFlag = True
            break  

    e = fl(0)
    p_tilde = np.copy(p)    # Allows for change to p_tilde without affecting p, as opposed to setting p_tilde = p

    for i in range(n):
        for j in range(K-1):
            sigmaPrime = sigma_tilde[j] + p_tilde[i]
            q = sigmaPrime - sigma_tilde[j]
            p_tilde[i] = p_tilde[i] - q
            sigma_tilde[j] = sigmaPrime
        e += p_tilde[i]
    
    if ExactFlag:
        res = T_values[0] + e
        return(res)
    
    t = fl(0)
    tau = [fl(0)]*(K-1)
    for i in range(K-1):
        tau[i] = sigma_tilde[i] - sigma_0[i]
        t_m = t + tau[i]     

        if abs(t_m) >= phi[i]:
            tau_2 = (t - t_m) + tau[i]
            if i == (K-2):
                res = t_m + (tau_2 + e)
            else:
                tau[i+1] = sigma_tilde[i+1] - sigma_0[i+1]
                tau_3, tau_4 = fasttwosum(tau_2, tau[i+1])
                # tau_3 = tau_2 + tau[i+1]
                # tau_4 = (tau_2 - tau_3) + tau[i+1]
                if i == (K-3):
                    res = t_m + (tau_3 + (tau_4 + e))
                else:
                    tau[i+2] = sigma_tilde[i+2] - sigma_0[i+2]
                    res = t_m + (tau_3 + (tau_4 + tau[i+2]))
            return (res)
        t = t_m
    res = t_m + e
    return(res)

# Define 5 numbers as float16
val = np.finfo(np.float16).tiny
num1 = np.float16(1.0)
num2 = np.float16((np.random.rand()*2-1)*val)   # Allows for values in the interval [-val, val]
num3 = np.float16((np.random.rand()*2-1)*val)
num4 = np.float16((np.random.rand()*2-1)*val)
num5 = np.float16((np.random.rand()*2-1)*val)

nums = [num1, num2, num3, num4, num5]
nums2= [num2, num3, num4, num5, num1]

# Summation order 1: Small numbers first, then the large number
sum_order1 = np.float16(0.0)
for num in [num2, num3, num4, num5, num1]:
    sum_order1 += num

# Summation order 2: Large number first, then the small numbers
sum_order2 = np.float16(0.0)
for num in [num1, num2, num3, num4, num5]:
    sum_order2 += num

# Summation using higher precision (numpy float32) for comparison
sum_high_precision = np.float32(num1) + np.float32(num2) + np.float32(num3) + np.float32(num4) + np.float32(num5)

print(f"Numbers (float16): {nums}")
print(f"Summation Order 1 (small numbers first): {sum_order1}")
print(f"Summation Order 2 (large number first): {sum_order2}")
print(f"Summation High Precision (float64): {sum_high_precision}")
print(f"Difference (Order 1 - High Precision): {abs(sum_order1 - sum_high_precision)}")
print(f"Difference (Order 2 - High Precision): {abs(sum_order2 - sum_high_precision)}")
print(f"Difference (Order 1 - Order 2): {abs(sum_order1 - sum_order2)}")
print("relative error (Order 1 - High Precision):", 100*abs(float(sum_order1) - sum_high_precision) / float(abs(sum_high_precision)), "%")
print("\n")

# FastPrecSum K = 2
fastprecsum = FastPrecSum(nums, 2)

print(f"Numbers (float16): {nums}")
print(f"FastPrecSum K = 2 (Order 1): {fastprecsum}")
print(f"High Precision (float32): {sum_high_precision}")
print(f"Difference (Order 1 - High Precision): {abs(fastprecsum - sum_high_precision)}")
print("relative error (Order 1 - High Precision):", 100*abs(float(fastprecsum) - sum_high_precision) / float(abs(sum_high_precision)), "%")
print("\n")

# FastPrecSum K = 3
fastprecsum2 = FastPrecSum(nums, 3)

print(f"Numbers (float16): {nums}")
print(f"FastPrecSum K = 3 (Order 1): {fastprecsum2}")
print(f"High Precision (float32): {sum_high_precision}")
print(f"Difference (Order 1 - High Precision): {abs(fastprecsum2 - sum_high_precision)}")
print("relative error (Order 1 - High Precision):", 100*abs(float(fastprecsum2) - sum_high_precision) / float(abs(sum_high_precision)), "%")
print("\n")

# FastPrecSum K = 4
fastprecsum3 = FastPrecSum(nums, 4)

print(f"Numbers (float16): {nums}")
print(f"FastPrecSum K = 4 (Order 1): {fastprecsum3}")
print(f"High Precision (float32): {sum_high_precision}")
print(f"Difference (Order 1 - High Precision): {abs(fastprecsum3 - sum_high_precision)}")
print("relative error (Order 1 - High Precision):", 100*abs(float(fastprecsum3) - sum_high_precision) / float(abs(sum_high_precision)), "%")
print("\n")

# FastPrecSum K = 5
fastprecsum4 = FastPrecSum(nums, 5)

print(f"Numbers (float16): {nums}")
print(f"FastPrecSum K = 5 (Order 1): {fastprecsum4}")
print(f"High Precision (float32): {sum_high_precision}")
print(f"Difference (Order 1 - High Precision): {abs(fastprecsum4 - sum_high_precision)}")
print("relative error (Order 1 - High Precision):", 100*abs(float(fastprecsum4) - sum_high_precision) / float(abs(sum_high_precision)), "%")
