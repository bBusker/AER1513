import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import scipy.io
from sympy import Matrix, symbols, init_printing, cos, sin, sqrt, atan2, latex, diag, eye
from sympy.utilities.lambdify import lambdify
init_printing()


def wraptopi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def run(printing=True, EKF=True):
    A, B = symbols("A B")
    x_k, y_k, theta_k = symbols("x_k y_k theta_k")
    x_kp, y_kp, theta_kp = symbols("x_{k-1} y_{k-1} \\theta_{k-1}")
    x_l, y_l = symbols("x_l y_l")
    v_k, omega_k = symbols("v_k omega_k")
    T = symbols("T")
    d = symbols ("d")
    noise_v, noise_omega = symbols("n_v, n_omega")
    var_v, var_omega = symbols("sigma_v^2, sigma_omega^2")
    noise_r, noise_phi = symbols("n_r, n_phi")
    var_r, var_phi = symbols("sigma_r^2, sigma_phi^2")
    noises = (noise_v, noise_omega, noise_r, noise_phi)

    X_k = Matrix([x_k, y_k, theta_k])
    X_kp = Matrix([x_kp, y_kp, theta_kp])
    AA = Matrix([[cos(theta_kp), 0], [sin(theta_kp), 0], [0, 1]])
    U_k = Matrix([v_k, omega_k])
    W_k = Matrix([noise_v, noise_omega])
    h = X_kp + T*AA*(U_k + W_k)

    N_k = Matrix([noise_r, noise_phi])
    g = Matrix([sqrt((x_l - x_k - d*cos(theta_k))**2 + (y_l - y_k - d*sin(theta_k))**2),
                atan2(y_l - y_k - d*sin(theta_k), x_l - x_k - d*cos(theta_k)) - theta_k]) + N_k

    F_kp = h.jacobian(X_kp).subs([(noise, 0) for noise in noises])
    Wp_k = h.jacobian(W_k).subs([(noise, 0) for noise in noises])*W_k
    Qp_k = Wp_k * Wp_k.T
    # Manually do expectation
    Qp_k = Qp_k.subs([(noise_omega*noise_v, 0), (noise_v**2, var_v), (noise_omega**2, var_omega)])


    G_k = g.jacobian(X_k).subs([(noise, 0) for noise in noises])
    G_k_print = G_k.subs([(d*cos(theta_k)+x_k-x_l, A), (d*sin(theta_k)+y_k-y_l, B)])
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
        print(latex(G_k_print))
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
    r_max = 5
    save_name_suffix = "_badinit_r5"

    P_prior = diag(1, 1, 0.1)
    # X_prior = Matrix([x_true[0], y_true[0], theta_true[0]])  # Good init
    X_prior = Matrix([1, 1, 0.1])  # Bad init
    Xs = [np.array(X_prior).astype(np.float64)]
    Ps = [np.array(P_prior).astype(np.float64)]

    F_kp_l = lambdify([v_k, theta_kp, T], F_kp, 'numpy')
    Qp_k_l = lambdify([var_v, var_omega, theta_kp, T], Qp_k, 'numpy')
    h_l = lambdify([x_kp, y_kp, theta_kp, v_k, omega_k, noise_v, noise_omega, T], h, 'numpy')
    G_k_l = lambdify([x_k, y_k, theta_k, x_l, y_l, d], G_k, 'numpy')
    g_l = lambdify([x_k, y_k, theta_k, x_l, y_l, noise_r, noise_phi, d], g, 'numpy')


    print("Starting EKF...")
    for i in range(x_true.shape[0]):
        F_kp_ = F_kp_l(v[i][0], float(X_prior[2]), 0.1)
        Qp_k_ = Qp_k_l(v_var.item(), omega_var.item(), float(X_prior[2]), 0.1)

        P_post = F_kp_ * P_prior * F_kp_.T + Qp_k_
        X_post = h_l(float(X_prior[0]), float(X_prior[1]), float(X_prior[2]), v[i][0], omega[i][0], 0, 0, 0.1)

        Gs = []
        gs = []
        ys = []
        valid_landmarks = 0
        for j in range(r.shape[1]):
            if r[i][j] == 0 or r[i][j] > r_max: continue
            valid_landmarks += 1
            G = G_k_l(float(X_post[0]), float(X_post[1]), float(X_post[2]), landmarks[j][0], landmarks[j][1], d_.item())
            g_ = g_l(float(X_post[0]), float(X_post[1]), float(X_post[2]), landmarks[j][0], landmarks[j][1], 0, 0, d_.item())
            g_[1] = wraptopi(g_[1])

            Gs.append(G)
            gs.append(g_)
            ys.append(np.array([[r[i][j], b[i][j]]]).T)

        if valid_landmarks != 0:
            G_ = np.vstack(Gs)
            g_ = np.vstack(gs)
            y_ = np.vstack(ys)
            R_ = np.diag([r_var.item(), b_var.item()]*valid_landmarks)

            K = P_post * G_.T * (G_ * P_post * G_.T + R_).inv()
            P_prior = (np.eye(3) - K * G_) * P_post
            X_prior = X_post + K * (y_ - g_)
        else:
            P_prior = P_post
            X_prior = X_post

        state = np.array([x_true[i], y_true[i], theta_true[i]])
        state_err = X_prior - state
        print(f"\rFinished {i}/{x_true.shape[0]} xy err: {(abs(state_err[0]) + abs(state_err[1]))/2}")
        Xs.append(np.array(X_prior).astype(np.float64))
        Ps.append(np.array(P_prior).astype(np.float64))

    Xs = np.array([Xs]).T
    Ps = np.array(Ps)
    np.save(f"Xs{save_name_suffix}", Xs)
    np.save(f"Ps{save_name_suffix}", Ps)
    err = np.abs(x_true - Xs.squeeze()[:, 1:].T)
    print(f"Avg error: {np.average(err)}")


def plotting():
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

    for r_max in [1, 3, 5]:
        filename = f'r{r_max}'
        res = np.load(f'Xs_{filename}.npy').squeeze()
        var = np.load(f'Ps_{filename}.npy').squeeze()
        stddv = np.sqrt(var)

        fig, ax = plt.subplots(3, 1, figsize=(5, 10))
        err_x = res[0, 1:] - x_true.squeeze()
        ax[0].plot(t.squeeze(), err_x, linewidth=0.3, label="Error in x")
        ax[0].fill_between(t.squeeze(), -3*stddv[1:, 0, 0], +3*stddv[1:, 0, 0], edgecolor='#CC4F1B', facecolor='#FF9848', alpha=0.5, linestyle=':', label='Uncertainty Envelope')
        err_y = res[1, 1:] - y_true.squeeze()
        ax[1].plot(t.squeeze(), err_y, linewidth=0.3, label="Error in y")
        ax[1].fill_between(t.squeeze(), -3*stddv[1:, 1, 1], +3*stddv[1:, 1, 1], edgecolor='#CC4F1B', facecolor='#FF9848', alpha=0.5, linestyle=':', label='Uncertainty Envelope')
        # err_theta = np.abs(res[2, 1:] - theta_true.squeeze())
        theta_diff = res[2, 1:] - theta_true.squeeze()
        err_theta = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
        ax[2].plot(t.squeeze(), err_theta, linewidth=0.3, label="Error in theta")
        ax[2].fill_between(t.squeeze(), -3*stddv[1:, 2, 2], +3*stddv[1:, 2, 2], edgecolor='#CC4F1B', facecolor='#FF9848', alpha=0.5, linestyle=':', label='Uncertainty Envelope')
        fig.suptitle(f"Errors for R={r_max}")
        ax[0].set_title("Errors for x")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Error (m)")
        ax[0].legend()
        ax[1].set_title("Errors for y")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Error (m)")
        ax[1].legend()
        ax[2].set_title("Errors for theta")
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("Error (rad)")
        ax[2].legend()
        fig.show()
        fig.savefig(f"{filename}.png")


if __name__ == "__main__":
    run(printing=True, EKF=True)
    # plotting()
