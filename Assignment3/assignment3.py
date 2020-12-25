import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def get_initial_guess(k1=1215, k2=1714, T_gt=None):
    if T_gt is None:
        C_gt = aa_to_C(theta_vk_i[:, k1])
        r_gt = - C_gt @ r_i_vk_i[:, k1]
        T_gt = Tmat(C_gt, r_gt)

    Ts = np.zeros((t.shape[1], 4, 4))
    Ts[k1] = T_gt
    for k in range(k1+1, k2+1):
        Tk = t[0][k] - t[0][k-1]
        # Rotation
        C_kp = getC(Ts[k-1])
        phi = wrap_to_pi(w_vk_vk_i[:, k-1] * Tk)
        dC = aa_to_C(phi)
        C_k = dC @ C_kp
        # Translation
        r_kp = -C_kp.T @ getR(Ts[k-1])
        d = v_vk_vk_i[:, k-1] * Tk
        dr = C_kp.T @ d
        r_k = r_kp + dr
        r_k = -C_k @ r_k
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

def gn_batch_estimate(T_op, k1=1215, k2=1714, iters=7):
    K = k2 - k1
    # T_op = get_ground_truth(k1, k2)


    for _ in range(iters):
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
        if len(e_y_ks) == 0:  # In case we have no valid measurements
            e = e_top
        else:
            e_bot = np.vstack(e_y_ks)
            e = np.vstack((e_top, e_bot))
        print(f"e_top: {np.average(np.abs(e_top))}")
        print(f"e_bot: {np.average(np.abs(e_bot))}")
        print(f"e_bot_max: {np.max(np.abs(e_bot))}")
        print(f"e_bot_argmax: {np.argmax(np.abs(e_bot))}")

        W = block_diag(*(Qs + Rs))
        Winv = W.copy()
        np.fill_diagonal(Winv, 1/W.diagonal())
        HTWinv = H.T @ Winv

        A = HTWinv @ H
        b = HTWinv @ e

        dx_opt = np.linalg.inv(A) @ b

        for k in range(K+1):
            T_op[k + k1] = expm(get_cross_op((iters-_)/iters * dx_opt[6*k:6*k+6])) @ T_op[k + k1]

        print(f"dx_opt: {np.average(np.abs(dx_opt))}")
        print("----------------------------------------")

    return T_op, A

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


def plot_errs(T_op, A, k1, k2):
    T_gt = get_ground_truth(k1, k2)

    rot_err = []
    trans_err = []
    for k in range(k1, k2+1):
        C_gt = getC(T_gt[k])
        C_op = getC(T_op[k])

        r_gt = getR(T_gt[k])
        r_op = getR(T_op[k])

        rot_err.append(get_inv_cross_op(np.eye(3) - C_op @ C_gt.T))
        trans_err.append(r_op - r_gt)
    rot_err = np.array(rot_err)
    trans_err = np.array(trans_err)

    print(f"Avg Rot Err: {np.average(np.abs(rot_err))}")
    print(f"Avg Trans Err: {np.average(np.abs(trans_err))}")

    var = np.linalg.inv(A).diagonal()
    var_tx = var[0::6]
    var_ty = var[1::6]
    var_tz = var[2::6]
    var_rx = var[3::6]
    var_ry = var[4::6]
    var_rz = var[5::6]

    var_ts = [var_tx, var_ty, var_tz]
    var_rs = [var_rx, var_ry, var_rz]


    t = np.arange(k1, k2+1)

    fig, ax = plt.subplots(3, 2, figsize=(10, 20))
    # fig.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    axis = ["x", "y", "z"]
    for i in range(3):
        ax[i][0].plot(t, trans_err[:, i], label="Translation Error")
        ax[i][0].fill_between(t, -3 * np.sqrt(var_ts[i]), +3 * np.sqrt(var_ts[i]), edgecolor='#CC4F1B',
                           facecolor='#FF9848', alpha=0.5, linestyle=':', label='Uncertainty Envelope')
        ax[i][0].set_xlabel("Timestep")
        ax[i][0].set_ylabel("Error [m]")
        ax[i][0].set_title(f"Translation Error in {axis[i]} Axis")
        ax[i][0].legend()

        ax[i][1].plot(t, rot_err[:, i], label="Rotation Error")
        ax[i][1].fill_between(t, -3 * np.sqrt(var_rs[i]), +3 * np.sqrt(var_rs[i]), edgecolor='#CC4F1B',
                           facecolor='#FF9848', alpha=0.5, linestyle=':', label='Uncertainty Envelope')
        ax[i][1].set_xlabel("Timestep")
        ax[i][1].set_ylabel("Error [rad]")
        ax[i][1].set_title(f"Rotation Error in {axis[i]} Axis")
        ax[i][1].legend()
    plt.show()

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

    fig, ax = plt.subplots(figsize=(15, 4))
    fig.suptitle('Visible Landmarks at Each Timestep for Timesteps 1215-1714')
    ax.scatter(np.arange(1215, 1715), num_valids[1215:1715], s=0.5, c=colors[1215:1715])
    ax.plot(np.arange(1215, 1715), num_valids[1215:1715], linewidth=0.1, c='k')
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("Number of Visible Landmarks")
    ax.set_xlabel("Timestep Number")
    plt.savefig("q4_zoomed.png", bbox_inches='tight')

def q5a():
    k1, k2 = 1215, 1714
    T_op = get_initial_guess(k1, k2)
    T_opt, A = gn_batch_estimate(T_op, k1, k2, iters=10)
    plot_errs(T_opt, A, k1, k2)

def q5b():
    k1, k2 = 1215, 1714
    kappa = 50

    T_op = get_initial_guess(k1, k1+kappa)
    T_opt, A = gn_batch_estimate(T_op, k1, k1+kappa, iters=10)

    Ts = np.zeros_like(T_opt)
    As = np.zeros(((k2-k1+1)*6, (k2-k1+1)*6))
    Ts[k1] = T_opt[k1]
    var = np.linalg.inv(A).diagonal()
    As[0:6, 0:6] = np.linalg.inv(np.diag(var[0:6]))

    for k in range(k1+1, k2+1):
        print(f"Current timestep: {k}")
        T_op = get_initial_guess(k, k+kappa, T_gt=Ts[k-1])
        T_opt, A = gn_batch_estimate(T_op, k, k+kappa, iters=10)
        Ts[k] = T_opt[k]
        var = np.linalg.inv(A).diagonal()
        As[(k-k1)*6:(k-k1)*6+6, (k-k1)*6:(k-k1)*6+6] = np.linalg.inv(np.diag(var[0:6]))

    plot_errs(Ts, As, k1, k2)

def q5c():
    k1, k2 = 1215, 1714
    kappa = 10

    T_op = get_initial_guess(k1, k1+kappa)
    T_opt, A = gn_batch_estimate(T_op, k1, k1+kappa, iters=10)

    Ts = np.zeros_like(T_opt)
    As = np.zeros(((k2-k1+1)*6, (k2-k1+1)*6))
    Ts[k1] = T_opt[k1]
    var = np.linalg.inv(A).diagonal()
    As[0:6, 0:6] = np.linalg.inv(np.diag(var[0:6]))

    for k in range(k1+1, k2+1):
        print(f"Current timestep: {k}")
        T_op = get_initial_guess(k, k+kappa, T_gt=Ts[k-1])
        T_opt, A = gn_batch_estimate(T_op, k, k+kappa, iters=10)
        Ts[k] = T_opt[k]
        var = np.linalg.inv(A).diagonal()
        As[(k-k1)*6:(k-k1)*6+6, (k-k1)*6:(k-k1)*6+6] = np.linalg.inv(np.diag(var[0:6]))

    plot_errs(Ts, As, k1, k2)

if __name__ == "__main__":
    q4()
    q5a()
    q5b()
    q5c()