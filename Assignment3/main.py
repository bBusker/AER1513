import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import scipy.io
import math

from Assignment3.liegroups import SE3, SO3
from Assignment3.utils import *

# Load dataset data into variables
dataset = scipy.io.loadmat("dataset3.mat")
theta_vk_i = dataset['theta_vk_i']  # 3xK matrix where kth column is the axis-angle representation of the gt value of C_{v_k,i}. Used as phi_k in 3.3c to get the rotation matrix
r_i_vk_i = dataset['r_i_vk_i']  # 3xK matrix where the kth column is the gt value of r_i^{v_k,i} [m]
t = dataset['t']  # 1xK matrix of time values t(k) [s]
w_vk_vk_i = dataset['w_vk_vk_i']  # 3xK matrix where kth column is the measured rotational velocity w_{v_k}^{v_k,i} [rad/s]
w_var = dataset['w_var']  # 3x1 matrix of the computed variances of rotational speeds [rad^2/s^2]
v_vk_vk_i = dataset['v_vk_vk_i']  # 3xK matrix where the kth column is the measured translational velocity, v_{v_k}^{v_k,i} [m/s]
v_var = dataset['v_var']  # 3x1 matrix of the computed variances of translational speeds [m^2/s^2]
rho_i_pj_i = dataset['rho_i_pj_i']  # 3x20 matrix where jth column is the position of feature j [m]
y_k_j = dataset['y_k_j']  # 4xKx20 matrix of observations, y_k^j [pixels]. All components of y_k_j[:,k,j] will be -1 if the observation is invalid
y_var = dataset['y_var']  # 4x1 matrix of the computed variances of stereo measurements [pixels^2]
C_c_v = dataset['C_c_v']  # 3x3 matrix giving the rotation from the vehicle frame to the camera frame, C_{c,v}
rho_v_c_v = dataset['rho_v_c_v']  # 3x1 matrix giving the translation from the vehicle frame to the camera frame, rho_v^{c,v} [m]
fu = dataset['fu']  # stereo camera horizontal focal length [pixels]
fv = dataset['fv']  # stereo camera vertical focal length [pixels]
cu = dataset['cu']  # stereo camera vertical focal length [pixels]
cv = dataset['cv']  # stereo camrea horizontal optical center [pixels]
b = dataset['b']  # stereo camera baseline [m]

def q4():
    valids = np.where(y_k_j == -1, 0, 1)
    num_valids = valids[0, :, :].sum(-1)
    colors = np.array(['g' if num>=3 else 'r' for num in num_valids])
    t_sq = t.squeeze()

    fig, ax = plt.subplots(figsize=(15,4))
    fig.suptitle('Visible Landmarks at Each Timestep')
    ax.scatter(np.arange(0, t_sq.shape[0]), num_valids, s=0.5, c=colors)
    ax.plot(np.arange(0, t_sq.shape[0]), num_valids, linewidth=0.1, c='k')
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("Number of Visible Landmarks")
    ax.set_xlabel("Timestep Number")
    plt.savefig("q4.png", bbox_inches='tight')

def get_initial_guess(k1=1215, k2=1714):
    C_gt = aa_to_C(theta_vk_i[:, k1])
    r_gt = r_i_vk_i[:, k1]
    T_gt = Tmat(C_gt, r_gt)

    Ts = np.zeros((t.shape[1], 4, 4))
    Ts[k1] = T_gt
    for k in range(k1+1, k2+1):
        Tk = t[0][k] - t[0][k-1]
        # Rotation
        C_kp = getC(Ts[k-1])
        phi = w_vk_vk_i[:, k-1] * Tk
        dC = aa_to_C(phi)
        C_k = dC @ C_kp
        # Translation
        r_kp = getR(Ts[k-1])
        d = v_vk_vk_i[:, k-1] * Tk
        dr = C_kp.T @ d
        r_k = r_kp + dr
        # Transformation
        Ts[k] = Tmat(C_k, r_k)

    return Ts

def gn_batch_estimate(k1=1215, k2=1714):
    T_op = get_initial_guess(k1, k2)


    rs = []
    for i in range(T_op.shape[0]):
        T = T_op[i]
        if T[3][3] == 0: continue
        rs.append(getR(T))
    rs = np.array(rs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs=rs[:, 0], ys=rs[:, 1], zs=rs[:,2])
    plt.show()

    print("done")



if __name__ == "__main__":
    # q4()
    gn_batch_estimate()