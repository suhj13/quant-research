import numpy as np
import pandas as pd

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
        return(np.sum(np.abs(p)))                       # Return the regular sum value 

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

p = []
for i in range(6):
    p.append(fl(i+0.42245425764))

sum0 = 0
for i in p:
    sum0 += i
print("sum using regular loop:", sum0) 
print("sum with FastPrecSum:", FastPrecSum(p, 1))

sum = 0
for j in p:
    sum += np.float32(j)
print("sum in float32:", sum)
