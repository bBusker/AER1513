import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sympy import Matrix, symbols, init_printing, cos, sin, sqrt, atan2, latex, diag, eye
from sympy.utilities.lambdify import lambdify

# Load dataset data into variables
dataset = scipy.io.loadmat("dataset3.mat")
theta_vk_i = dataset['theta_vk_i']
r_i_vk_i = dataset['r_i_vk_i']
t = dataset['t']
w_vk_vk_i = dataset['w_vk_vk_i']
w_var = dataset['w_var']
v_vk_vk_i = dataset['v_vk_vk_i']
v_var = dataset['v_var']
rho_i_pj_i = dataset['rho_i_pj_i']
y_k_j = dataset['y_k_j']
y_var = dataset['y_var']
C_c_v = dataset['C_c_v']
rho_v_c_v = dataset['rho_v_c_v']
fu = dataset['fu']
cu = dataset['cu']
cv = dataset['cv']
b = dataset['b']

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

def gn_batch_estimate(k1=1215, k2=1714):
    pass



if __name__ == "__main__":
    q4()