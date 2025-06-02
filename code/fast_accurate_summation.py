import numpy as np

def fl(x):
    y = np.float16(x)
    return y

def fasttwosum(x,y):
    

def FastPrecSum(p, K):
    # p is an np array of floats
    # K is an integer representing the no. of fixed extractions
    n = len(p)
    ExactFlag = False

    eps = np.finfo(float).eps/2 # 2^-53
    eta = 2**(-1074)

    T_values = [0]*K
    T_values[0] = np.sum(np.abs(p))/(1-n*eps)
    
    sigma_0 = [0]*(K-1)
    sigma_tilde = [0]*(K-1)
    phi = [0]*(K-1)

    for i in range(K-1):
        sigma_0[i] = (2*T_values[i])/(1-(3*n+1)*eps)
        sigma_tilde[i] = sigma_0[i]

        q = (sigma_0[i]/(2*eps))
        u = np.abs(q/(1-eps)-q)   # u = ufp(sigma_0)

        phi[i] = ((2*n*(n+2)*eps)*u)/(1-5*eps)
        T_values[i+1] = min(((3/2 + 4*eps)*(n*eps)*sigma_0[i]),((2*n*eps)*u))

        if (4*T_values[i+1] <= (eta/eps)):
            K = i +1
            ExactFlag = True
            break  

    e = 0.0
    p_tilde = np.copy(p)    # Allows for change to p_tilde without affecting p, as opposed to setting p_tilde = p

    for i in range(n):
        for j in range(K-1):
            sigmaPrime = sigma_tilde[j] + p_tilde[i]
            q = sigmaPrime - sigma_tilde[j]
            p_tilde[i] = p_tilde[i] - q
            sigma_tilde[j] = sigmaPrime
        e += p_tilde[i]
    
    if ExactFlag:
        res = e
        return(res)
    
    t = 0.0
    tau = [0]*(K-1)
    for i in range(K-1):
        tau[i] = sigma_tilde[i] - sigma_0[i]
        t_m = t + tau[i]

        if abs(t_m) >= phi[i]:
            tau_2 = (t - t_m) + tau[i]
            if i == (K-2):
                res = t_m + (tau_2 + e)
                return(res)
            else:
                tau[i+1] = sigma_tilde[i+1] - sigma_0[i+1]
                tau_3 = tau_2 + tau[i+1]
                tau_4 = (tau_2 - tau_3) + tau[i+1]
                if i == (K-3):
                    res = t_m + (tau_3 + (tau_4 + e))
                else:
                    tau[i+2] = sigma_tilde[i+2] - sigma_0[i+2]
                    res = t_m + (tau_3 + (tau_4 + tau[i+2]))
            return (res)
        t = t_m
    res = t_m + e

n = 1000000
p = np.random.random(n)
K = 3

print(FastPrecSum(p, K))