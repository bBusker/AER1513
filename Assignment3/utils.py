import numpy as np
import math

def skewsym(v):
    if v.shape == (3,): v = v.reshape(3,1)
    if v.shape != (3,1): raise ValueError(f"Incorrect shape of phi: {v.shape}")
    return np.array([[0, -v[2][0], v[1][0]],
                     [v[2][0], 0, -v[0][0]],
                     [-v[1][0], v[0][0], 0]])

def aa_to_C(phi):
    if phi.shape == (3,): phi = phi.reshape(3,1)
    if phi.shape != (3,1): raise ValueError(f"Incorrect shape of phi: {phi.shape}")
    mag = np.linalg.norm(phi)
    unit = phi/mag
    return (math.cos(mag) * np.eye(3)) + ((1 - math.cos(mag)) * unit @ unit.T) - (math.sin(mag) * skewsym(unit))

def Tmat(C, r):
    if r.shape == (3,): r = r.reshape(3, 1)
    if C.shape != (3,3): raise ValueError(f"C matrix has incorrect dimensions {C.shape}")
    if r.shape != (3, 1): raise ValueError(f"R vector has incorrect dimensions {r.shape}")
    return np.vstack((np.hstack((C, r)), np.array([0, 0, 0, 1])))

def getC(T):
    return T[0:3, 0:3]

def getR(T):
    return T[0:3, 3]