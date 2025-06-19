import numpy as np 

def prob_RGB3(lambda_0, u):
    lambda_1 = np.sqrt(1-lambda_0**2)
    v = np.sqrt(1-u**2)

    U = [u,v]
    V = [v, -u]

    p_R_coarse = np.zeros((3, 3, 3))

    for i in range(2):
        p_R_coarse[0,0,i+1] = p_R_coarse[0,i+1,0] = p_R_coarse[i+1,0,0] = ((lambda_1**4) * (lambda_0**2) * (U[i]**2)) + ((lambda_0**4) * (lambda_1**2) * (V[i]**2))
        for j in range(2):
            for k in range(2):
                p_R_coarse[i+1, k+1, j+1] = ( ((lambda_1**3) * U[i] * U[j] * U[k]) + ((lambda_0**3) * V[i] * V[j] * V[k]) )**2
    return p_R_coarse

RGB3_specific = prob_RGB3(1 / np.sqrt(3), np.sqrt(0.215))




