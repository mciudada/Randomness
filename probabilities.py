import numpy as np 

def prob_RGB3(lambda_0, u):
    p_R_coarse = np.zeros((3, 3, 3))
    lambda_1 = np.sqrt(1-lambda_0**2)
    v = np.sqrt(1-u**2)
    p_R_coarse[0,0,1] = p_R_coarse[0,1,0] = p_R_coarse[1,0,0] = (lambda_1**4) * (lambda_0**2) * (u**2) + (lambda_0**4) * (lambda_1**2) * (v**2)
    p_R_coarse[0,0,2] = p_R_coarse[0,2,0] = p_R_coarse[2,0,0] = (lambda_1**4) * (lambda_0**2) * (v**2) + (lambda_0**4) * (lambda_1**2) * (u**2)

    p_R_coarse[1,1,2] = p_R_coarse[1,2,1] = p_R_coarse[2,1,1] = (lambda_1**3 * u**2 * v - lambda_0**3 * v**2 * u)**2
    p_R_coarse[1,2,2] = p_R_coarse[2,2,1] = p_R_coarse[2,1,2] = (lambda_1**3 * u * v**2 + lambda_0**3 * v * u**2)**2
    p_R_coarse[1,1,1] = (lambda_1**3 * u**3 +lambda_0**3 * v**3)**2
    p_R_coarse[2,2,2] = (lambda_1**3 * v**3 - lambda_0**3 * u**3)**2
    return p_R_coarse




