import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sympy import *
init_printing()

def run(printing=True):
    x_k, y_k, theta_k = symbols("x_k y_k theta_k")
    x_kp, y_kp, theta_kp = symbols("x_{k-1} y_{k-1} \\theta_{k-1}")
    x_l, y_l = symbols("x_l y_l")
    v_k, omega_k = symbols("v_k omega_k")
    T = symbols("T")
    d = symbols ("d")
    noise_v, noise_omega = symbols("noise_v, noise_omega")
    noise_r, noise_phi = symbols("noise_r, noise_phi")
    noises = (noise_v, noise_omega, noise_r, noise_phi)

    X_k = Matrix([x_k, y_k, theta_k])
    X_kp = Matrix([x_kp, y_kp, theta_kp])
    A = Matrix([[cos(theta_kp), 0], [sin(theta_kp), 0], [0, 1]])
    U_k = Matrix([v_k, omega_k])
    W_k = Matrix([noise_v, noise_omega])
    h = X_kp + T*A*(U_k + W_k)

    N_k = Matrix([noise_r, noise_phi])
    g = Matrix([sqrt((x_l - x_k - d*cos(theta_k))**2 + (y_l - y_k - d*sin(theta_k))**2),
                atan2(y_l - y_k - d*sin(theta_k), x_l - x_k - d*cos(theta_k)) - theta_k]) + N_k

    F_kp = h.jacobian(X_kp).subs([(noise, 0) for noise in noises])
    Wp_k = h.jacobian(W_k).subs([(noise, 0) for noise in noises])*W_k
    Qp_k = Wp_k * Wp_k.T


    G_k = g.jacobian(X_k).subs([(noise, 0) for noise in noises])
    Np_k = g.jacobian(N_k).subs([(noise, 0) for noise in noises])*N_k
    Rp_k = Np_k * Np_k.T

    if printing:
        print("F_k-1")
        print(latex(F_kp))
        print("W'_k")
        print(latex(Wp_k))
        print("G_k")
        print(latex(G_k))
        print("N'_k")
        print(latex(Np_k))
        print("Q'_k")
        print(latex(Qp_k))
        print("R'_k")
        print(latex(Rp_k))

    ############### EKF ###############
    '''
    F_kp: 3x3
    P_prior: 3x3
    Q'_k: 3x3
    P_post: 3x3
    x: 3x1
    G: (2*n)x3
    R: 2x2
    K: 3x(2*n)
    y: 2x1
    '''

    dataset = scipy.io.loadmat("dataset2.mat")
    t = dataset['t']
    x_true = dataset["x_true"]
    y_true = dataset["y_true"]
    theta_true = dataset["th_true"]
    true_valid = dataset["true_valid"]
    landmarks = dataset["l"]
    r = dataset["r"]
    r_var = dataset["r_var"]
    b = dataset["b"]
    b_var = dataset["b_var"]
    v = dataset["v"]
    v_var = dataset["v_var"]
    omega = dataset["om"]
    omega_var = dataset["om_var"]
    d_ = dataset["d"]

    P_prior = np.diag((1, 1, 0.1))
    X_prior = np.array([[x_true[0], y_true[0], theta_true[0]]]).T
    Xs = []

    for i in range(x_true.shape[0]):
        P_post = F_kp * P_prior * F_kp.T + Qp_k
        X_post = h.subs([(x_kp, X_prior[0][0]), (y_kp, X_prior[0][1]), (theta_kp, X_prior[0][2]), (v_k, v[i]), (omega_k, omega[i]), (noise_v, 0), (noise_omega, 0)])
        KG = Matrix([[0,0,0],[0,0,0],[0,0,0]])
        X_prior = X_post
        for j in range(17):
            if r[i][j] == 0: continue
            G = G_k.subs([(d, d_)])
            K = P_post * G.T * (G * P_post * G.T + Rp_k).inv()
            KG += K*G
            X_prior += K * (Matrix([r[i][j], b[i][j]]) - g.subs([(x_k, X_post[0][0]), (y_k, X_post[0][1]), (theta_k, X_post[0][2]), (x_l, landmarks[j][0]), (y_l, landmarks[j][1]), (noise_r, 0), (noise_phi, 0)]))
        P_prior = (eye(3) - KG) * P_post
        Xs.append(X_prior)


if __name__ == "__main__":
    run()

