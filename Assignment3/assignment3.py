import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import scipy.io
from scipy.linalg import expm, logm, block_diag
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
fu = dataset['fu'].item()  # stereo camera horizontal focal length [pixels]
fv = dataset['fv'].item()  # stereo camera vertical focal length [pixels]
cu = dataset['cu'].item()  # stereo camera horizontal optical center [pixels]
cv = dataset['cv'].item()  # stereo camrea vertical optical center [pixels]
b = dataset['b'].item()  # stereo camera baseline [m]

# Create transformation matrices
T_c_v = Tmat(C_c_v, -C_c_v @ rho_v_c_v)

# Create camera matrices
D = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
])
M = np.array([
    [fu, 0, cu, 0],
    [0, fv, cv, 0],
    [fu, 0, cu, -fu*b],
    [0, fv, cv, 0]
])
def dfdp(p):
    df = np.array([
        [1, 0, -p[0].item()/p[2].item(), 0],
        [0, 1, -p[1].item()/p[2].item(), 0],
        [0, 0, 0, 0],
        [0, 0, -p[3].item()/p[2].item(), 1]
    ])
    return M @ ((1/p[2].item())*df)


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
    r_gt = - C_gt @ r_i_vk_i[:, k1]
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

def get_ground_truth(k1=1215, k2=1714):
    Ts = np.zeros((t.shape[1], 4, 4))
    for k in range(k1, k2+1):
        C_gt = aa_to_C(theta_vk_i[:, k])
        r_gt = -C_gt @ r_i_vk_i[:, k]
        T_gt = Tmat(C_gt, r_gt)
        Ts[k] = T_gt
    return Ts

def gn_batch_estimate(k1=1215, k2=1714):
    K = k2 - k1
    T_op = get_initial_guess(k1, k2)
    # T_op = get_ground_truth(k1, k2)


    for _ in range(50):
        Fs = []
        Gs = []
        e_v_ks = []

        e_y_ks = []
        Qs = []
        Rs = []

        # Error for timestep 0
        C_gt = aa_to_C(theta_vk_i[:, k1])
        r_gt = -C_gt @ r_i_vk_i[:, k1]
        T_gt = Tmat(C_gt, r_gt)
        e_v_ks.append(get_inv_cross_op(logm(T_gt @ np.linalg.inv(T_op[k1]))))

        for k in range(k1+1, k2+1):
            dt = t[0][k] - t[0][k-1]
            T_op_k = T_op[k]
            T_op_kp = T_op[k - 1]
            omega_k = np.hstack((-v_vk_vk_i[:, k], -w_vk_vk_i[:, k])).reshape((6, 1))

            xi_k = expm(dt * get_cross_op(omega_k))
            e_v_k = get_inv_cross_op(logm(xi_k @ T_op_kp @ np.linalg.inv(T_op_k)))
            e_v_ks.append(e_v_k)

            F_kp = get_Ad(T_op_k @ np.linalg.inv(T_op_kp))
            Fs.append(F_kp)

        for k in range(k1, k2+1):
            dt = t[0][k] - t[0][k-1]
            T_op_k = T_op[k]

            G_ks = []
            e_ys = []
            for j in range(20):
                y_j = y_k_j[:, k, j].reshape((4, 1))
                if y_j[0] == -1: continue
                p_j = np.vstack((rho_i_pj_i[:, j].reshape((3,1)), np.eye(1)))
                p_j_c = T_c_v @ T_op_k @ p_j
                G_ks.append(dfdp(p_j_c) @ T_c_v @ get_circ_op(T_op_k @ p_j))
                e_ys.append(y_j - M @ (p_j_c/p_j_c[2].item()))

            if G_ks:  # If we have valid observations/list is not empty
                Gk = np.vstack(G_ks)
                e_y_k = np.vstack(e_ys)

                Gs.append(Gk)
                e_y_ks.append(e_y_k)
            else:
                Gs.append(np.zeros((0,6)))

            Qs.append(dt**2 * np.diag(np.vstack((v_var, w_var)).squeeze()))
            for _ in range(len(G_ks)):
                Rs.append(np.diag(y_var.squeeze()))

        H_top = np.eye(K * 6 + 6)
        for i in range(len(Fs)):
            H_top[6*i+6:6*i+12, 6*i:6*i+6] = -Fs[i]
        H_bot = block_diag(*Gs)
        H = np.vstack((H_top, H_bot))

        e_top = np.vstack(e_v_ks)
        e_bot = np.vstack(e_y_ks)
        print(f"e_top: {np.average(np.abs(e_top))}")
        print(f"e_bot: {np.average(np.abs(e_bot))}")
        print(f"e_bot_max: {np.max(np.abs(e_bot))}")
        print(f"e_bot_argmax: {np.argmax(np.abs(e_bot))}")
        e = np.vstack((e_top, e_bot))

        W = block_diag(*(Qs + Rs))
        Winv = W.copy()
        np.fill_diagonal(Winv, 1/W.diagonal())
        HTWinv = H.T @ Winv

        A = HTWinv @ H
        b = HTWinv @ e

        dx_opt = np.linalg.inv(A) @ b

        for k in range(K+1):
            T_op[k + k1] = expm(get_cross_op(dx_opt[6*k:6*k+6])) @ T_op[k + k1]

        print(f"dx_opt: {np.average(np.abs(dx_opt))}")
        print("----------------------------------------")


def plot_figure(T_op):
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
    q4()
    gn_batch_estimate(1215, 1714)
    # gn_batch_estimate(1215, 1400)