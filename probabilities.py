import numpy as np 
from qutip import *

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

def prob_MNN(theta): # note that theta cannot be 0, pi/4, pi/2
    p=np.zeros(shape=(2,2,2,2,2)) # initialization of the probability
    
    phi = bell_state(state='00') #this is |phi+>
    rho1 = ket2dm(phi) # this is phi+ = |phi+><phi+|
    rho2 = rho1
    rho = tensor(rho1,rho2)
    
    A=np.zeros(shape=(2,2,2,2),dtype=np.complex128) # initialization of an hypermatrix A the last to indices corresponds to x and a respectively, i.e. A[:,:,0,1] is the measurement done when x=0 and a=-1 (en el ultimo indice 0 corresponde a 1 y 1 a -1)
    C=np.zeros(shape=(2,2,2,2),dtype=np.complex128) # the same for C
    
    A0 = sigmax() # measurement of A for x = 0
    A1 = sigmaz() # measurement of A for x = 1
    eival_A0,eivect_A0 = A0.eigenstates() #eigen of A for x = 0 
    eival_A1,eivect_A1 = A1.eigenstates() #eigen of A for x = 1
    
    A[:,:,0,0] = np.dot(eivect_A0[0],eivect_A0[0].dag()).full() # measurement of A for x = 0 corresponding to a=0
    A[:,:,0,1] = np.dot(eivect_A0[1],eivect_A0[1].dag()).full() # measurement of A for x = 0 corresponding to a=1
    A[:,:,1,0] = np.dot(eivect_A1[0],eivect_A1[0].dag()).full()
    A[:,:,1,1] = np.dot(eivect_A1[1],eivect_A1[1].dag()).full()
    
    C0 = sigmax()
    C1 = sigmaz()
    eival_C0,eivect_C0 = C0.eigenstates() #eigen of C for x = 0 
    eival_C1,eivect_C1 = C1.eigenstates() #eigen of C for x = 0
    
    C[:,:,0,0] = np.dot(eivect_C0[0],eivect_C0[0].dag()).full() # measurement of C for x = 0 corresponding to c=0
    C[:,:,0,1] = np.dot(eivect_C0[1],eivect_C0[1].dag()).full() # measurement of C for x = 0 corresponding to c=1
    C[:,:,1,0] = np.dot(eivect_C1[0],eivect_C1[0].dag()).full()
    C[:,:,1,1] = np.dot(eivect_C1[1],eivect_C1[1].dag()).full()
    
    B=np.zeros(shape=(4,4,2),dtype=np.complex128) #initialization for B (the last two indeces corresponds to b_0 and b_1
    
    psi_theta = np.sin(theta) * basis(4,1) + np.cos(theta) * basis(4,2) 
    
    B[:,:,0] = ket2dm(psi_theta).full() 
    B[:,:,1] = ((qeye(4) - ket2dm(psi_theta))).full()

    for a, b, c, x, z in np.ndindex(2, 2, 2, 2, 2):
        p[a,b,c,x,z] = np.trace(np.dot(rho.full(),np.kron(A[:,:,x,a],np.kron(B[:,:,b],C[:,:,z,c]))))
    
    return p

MNN_specific = prob_MNN(np.pi/8)

def prob_postquantum(E1, E2, E3):
    p = np.zeros((2, 2, 2, 2, 1, 1, 1, 1))
    for a, b, c in np.ndindex(2, 2, 2):
        ap = 2*a-1
        bp = 2*b-1
        cp = 2*c-1
        p[a, b, c, 0] = (1/8) * (1 
                        + (ap + bp + cp)*E1 
                        + (ap*bp + bp*cp + cp*ap)*E2 
                        + ap*bp*cp*E3)
    return p

def prob_ent(): # Entanglement-swapping
    p=np.zeros((2,2,2,2,2))
    
    phi = bell_state(state='00') 
    rho1 = ket2dm(phi) 
    rho2 = rho1
    rho = tensor(rho1,rho2)
    
    A=np.zeros((2, 2, 2, 2),dtype = 'complex_') 
    C=np.zeros(16,dtype = 'complex_').reshape([2,2,2,2]) 
    
    A0 = (1/np.sqrt(2))*(sigmax() - sigmaz()) 
    A1 = (1/np.sqrt(2))*(sigmax() + sigmaz()) 
    eival_A0,eivect_A0 = A0.eigenstates() 
    eival_A1,eivect_A1 = A1.eigenstates() 
    
    A[:,:,0,0] = np.dot(eivect_A0[0].full(),eivect_A0[0].dag().full()) 
    A[:,:,0,1] = np.dot(eivect_A0[1].full(),eivect_A0[1].dag().full()) 
    A[:,:,1,0] = np.dot(eivect_A1[0].full(),eivect_A1[0].dag().full())
    A[:,:,1,1] = np.dot(eivect_A1[1].full(),eivect_A1[1].dag().full())
    
    C0 = sigmax()
    C1 = sigmaz()
    eival_C0,eivect_C0 = C0.eigenstates() 
    eival_C1,eivect_C1 = C1.eigenstates() 
    
    C[:,:,0,0] = np.dot(eivect_C0[0].full(),eivect_C0[0].dag().full()) 
    C[:,:,0,1] = np.dot(eivect_C0[1].full(),eivect_C0[1].dag().full()) 
    C[:,:,1,0] = np.dot(eivect_C1[0].full(),eivect_C1[0].dag().full())
    C[:,:,1,1] = np.dot(eivect_C1[1].full(),eivect_C1[1].dag().full())
    
    B=np.zeros(32,dtype = 'complex_').reshape([4,4,2]) 
    
    phi_ = bell_state(state='01')
    psi = bell_state(state='10') 
    psi_ = bell_state(state='11')
    
    B[:,:,0] = ket2dm(psi).full()
    B[:,:,1] = (ket2dm(phi) + ket2dm(phi_) + ket2dm(psi_)).full()
    
    for a, b, c, x, z in np.ndindex(2, 2, 2, 2, 2):
        p[a,b,c,x,z] = np.trace(np.dot(rho.full(),np.kron(A[:,:,x,a],np.kron(B[:,:,b],C[:,:,z,c]))))
    
    return p


def prob_Fritz():
    p = np.zeros((2,2,2,2,2))
    phi = bell_state(state='00') 
    rho = ket2dm(phi) 
    
    A=np.zeros(16,dtype = 'complex_').reshape([2,2,2,2])
    B=np.zeros(16,dtype = 'complex_').reshape([2,2,2,2]) 
    
    A0 = sigmax()
    A1 = sigmaz()
    eival_A0,eivect_A0 = A0.eigenstates() 
    eival_A1,eivect_A1 = A1.eigenstates() 

    A[:,:,0,0] = np.dot(eivect_A0[0].full(),eivect_A0[0].dag().full()) 
    A[:,:,0,1] = np.dot(eivect_A0[1].full(),eivect_A0[1].dag().full()) 
    A[:,:,1,0] = np.dot(eivect_A1[0].full(),eivect_A1[0].dag().full())
    A[:,:,1,1] = np.dot(eivect_A1[1].full(),eivect_A1[1].dag().full())
    
    
    B0 = (1/np.sqrt(2))*(sigmax() - sigmaz())
    B1 = (1/np.sqrt(2))*(sigmax() + sigmaz())
    eival_B0,eivect_B0 = B0.eigenstates() 
    eival_B1,eivect_B1 = B1.eigenstates() 
    
    B[:,:,0,0] = np.dot(eivect_B0[0].full(),eivect_B0[0].dag().full()) 
    B[:,:,0,1] = np.dot(eivect_B0[1].full(),eivect_B0[1].dag().full()) 
    B[:,:,1,0] = np.dot(eivect_B1[0].full(),eivect_B1[0].dag().full())
    B[:,:,1,1] = np.dot(eivect_B1[1].full(),eivect_B1[1].dag().full())
    
    pc = np.zeros((2,2))
    for c, z in np.ndindex(2, 2):
        pc[c,z] = 1/2
        
    for a, b, c, x, z in np.ndindex(2, 2, 2, 2, 2):
        p[a,b,c,x,z] = pc[c,z] * np.trace(np.dot(rho.full(),np.kron(A[:,:,x,a],B[:,:,c,b])))
    
    return p



