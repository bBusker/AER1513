import math

from matplotlib import pyplot as plt
from scipy.linalg import expm, logm
from numpy.linalg import inv, norm
from numpy import cos, sin
import numpy as np

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

def get_cross_op(a):
    assert a.shape == (3, 1) or a.shape == (6, 1)
    if a.shape == (3, 1):
        a_cross = np.zeros((3, 3))
        a_cross[0, 1] = -a[2]
        a_cross[0, 2] = a[1]
        a_cross[1, 0] = a[2]
        a_cross[1, 2] = -a[0]
        a_cross[2, 0] = -a[1]
        a_cross[2, 1] = a[0]
    elif a.shape == (6, 1):
        a_cross = np.zeros((4, 4))
        a_cross[0, 1] = -a[5]
        a_cross[0, 2] = a[4]
        a_cross[1, 0] = a[5]
        a_cross[1, 2] = -a[3]
        a_cross[2, 0] = -a[4]
        a_cross[2, 1] = a[3]
        a_cross[0, 3] = a[0]
        a_cross[1, 3] = a[1]
        a_cross[2, 3] = a[2]
    return a_cross

def get_inv_cross_op(A):
    assert A.shape == (3, 3) or A.shape == (4, 4)
    if A.shape == (3, 3):
        a = np.zeros((3, 1))
        a[0] = A[2, 1]
        a[1] = A[0, 2]
        a[2] = A[1, 0]
    elif A.shape == (4, 4):
        a = np.zeros((6, 1))
        a[0] = A[0, 3]
        a[1] = A[1, 3]
        a[2] = A[2, 3]
        a[3] = A[2, 1]
        a[4] = A[0, 2]
        a[5] = A[1, 0]
    return a

def get_circ_op(p):
    assert p.shape == (4, 1)
    eps = p[:-1]
    eta = p[3, 0]
    A = np.zeros((4, 6))
    A[:3, :3] = eta*np.eye(3)
    A[:3, 3:] = -get_cross_op(eps)
    return A

def get_Ad(T):
    assert T.shape == (4, 4)
    ad_T = np.zeros((6, 6))
    C = T[:3, :3]
    r = T[:3, 3]
    ad_T[:3, :3] = C
    ad_T[3:, 3:] = C
    ad_T[:3, 3:] = get_cross_op(r.reshape((3, 1)))@C
    return ad_T

def psi_2_rot(psi):
    assert psi.shape == (3, 1)
    psi_norm = norm(psi)
    term1 = cos(psi_norm)*np.eye(3)
    term2 = (1-cos(psi_norm))*(psi/psi_norm)@(psi/psi_norm).T
    term3 = -sin(psi_norm)*get_cross_op(psi/psi_norm)
    return term1+term2+term3

def augment(x):
    assert x.shape == (3, 1)
    x = np.vstack((x, np.array([[1]])))
    return x

def show_sparsity_pattern(C):
    plt.spy(C)
    plt.show()

def wrap_to_pi(ang):
    ang = (ang+np.pi)%(2*np.pi)-np.pi
    return ang

if __name__ == "__main__":
    a = np.array([[1],[2],[3]])
    a_cross = get_cross_op(a)
    # print(a_cross)
    a = get_inv_cross_op(a_cross)
    # print(a)
    a = np.array([[4],[5],[6],[1],[2],[3]])
    a_cross = get_cross_op(a)
    # print(a_cross)
    a = get_inv_cross_op(a_cross)
    # print(a)
    a = np.array([[1],[2],[3]])
    C = psi_2_rot(a)
    # print(C)
    temp = np.eye(3)
    temp[-1,-1] = 100
    T = np.hstack((temp, np.array([[1],[2],[3]])))
    T = np.vstack((T, np.array([0,0,0,1])))
    ad_T = get_Ad(T)
    # print(ad_T)
    p = np.array([[4],[3],[2],[1]])
    A = get_circ_op(p)
    # print(A)

    C = psi_2_rot(a)
    T = np.hstack((C, np.array([[1],[2],[3]])))
    T = np.vstack((T, np.array([0,0,0,1])))
    T_inv = np.linalg.inv(T)
    e = get_inv_cross_op(logm(np.eye(4)))
    print(e)

    # a = np.array([[1],[2],[3],[4],[5],[6]])
    # z = np.array([[4],[3],[2],[1]])
    # a_cross = get_cross_op(a)
    # print(a_cross@z)
    # print(get_circ_op(z)@a)

