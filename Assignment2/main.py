import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sympy import Matrix, symbols, init_printing, cos, sin, sqrt, atan2, latex, diag, eye
from sympy.utilities.lambdify import lambdify
init_printing()


def wraptopi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def run(printing=True, EKF=True):
    x_k, y_k, theta_k = symbols("x_k y_k theta_k")
    x_kp, y_kp, theta_kp = symbols("x_{k-1} y_{k-1} \\theta_{k-1}")
    x_l, y_l = symbols("x_l y_l")
    v_k, omega_k = symbols("v_k omega_k")
    T = symbols("T")
    d = symbols ("d")
    noise_v, noise_omega = symbols("noise_v, noise_omega")
    var_v, var_omega = symbols("sigma_v, sigma_omega")
    noise_r, noise_phi = symbols("noise_r, noise_phi")
    var_r, var_phi = symbols("sigma_r, sigma_phi")
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
    # Manually do expectation
    Qp_k = Qp_k.subs([(noise_omega*noise_v, 0), (noise_v**2, var_v), (noise_omega**2, var_omega)])


    G_k = g.jacobian(X_k).subs([(noise, 0) for noise in noises])
    Np_k = g.jacobian(N_k).subs([(noise, 0) for noise in noises])*N_k
    Rp_k = Np_k * Np_k.T
    # Manually do expectation
    Rp_k = Rp_k.subs([(noise_phi*noise_r, 0), (noise_r**2, var_r), (noise_phi**2, var_phi)])

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
    if not EKF:
        return
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

    P_prior = diag(1, 1, 0.1)
    X_prior = Matrix([x_true[0], y_true[0], theta_true[0]])
    Xs = []

    F_kp_l = lambdify([v_k, theta_kp, T], F_kp, 'numpy')
    Qp_k_l = lambdify([var_v, var_omega, theta_kp, T], Qp_k, 'numpy')
    Rp_k_l = lambdify([var_r, var_phi], Rp_k, 'numpy')
    h_l = lambdify([x_kp, y_kp, theta_kp, v_k, omega_k, noise_v, noise_omega, T], h, 'numpy')
    G_k_l = lambdify([x_k, y_k, theta_k, x_l, y_l, d], G_k, 'numpy')
    g_l = lambdify([x_k, y_k, theta_k, x_l, y_l, noise_r, noise_phi, d], g, 'numpy')


    print("Starting EKF...")
    for i in range(x_true.shape[0]):
        print(f"\rFinished {i}/{x_true.shape[0]}", end="")
        F_kp_ = F_kp_l(v[i][0], float(X_prior[2]), 0.1)
        Qp_k_ = Qp_k_l(v_var.item(), omega_var.item(), float(X_prior[2]), 0.1)
        Rp_k_ = Rp_k_l(r_var.item(), b_var.item())

        P_post = F_kp_ * P_prior * F_kp_.T + Qp_k_
        X_post = h_l(float(X_prior[0]), float(X_prior[1]), float(X_prior[2]), v[i][0], omega[i][0], 0, 0, 0.1)

        KG = Matrix([[0,0,0],[0,0,0],[0,0,0]])
        X_prior = X_post
        for j in range(r.shape[1]):
            if r[i][j] == 0: continue
            G = G_k_l(float(X_post[0]), float(X_post[1]), float(X_post[2]), landmarks[j][0], landmarks[j][1], d_.item())
            K = P_post * G.T * (G * P_post * G.T + Rp_k_).inv()

            KG += K*G
            g_ = g_l(float(X_post[0]), float(X_post[1]), float(X_post[2]), landmarks[j][0], landmarks[j][1], 0, 0, d_.item())
            g_[1] = wraptopi(g_[1])
            X_prior += K * (Matrix([r[i][j], b[i][j]]) - g_)
            # print(X_prior.evalf())
        P_prior = (eye(3) - KG) * P_post
        print(X_prior)
        Xs.append(X_prior)

    Xs = np.array([Xs]).T
    err = np.abs(x_true - Xs)
    print(np.avg(err))

if __name__ == "__main__":
    run(printing=True, EKF=True)

