import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def Get_section_component(corde, r_section, S_ref, title, Cn_section):

    m = np.linspace(0.1, 3.0, 950)

    h = corde / (np.sqrt(3) * m)
    S = h * (2 * corde - h * np.tan(np.deg2rad(30)))
    idx_S_max = np.argmin(np.abs(S - S.max()))
    idx_S_0 = np.argmin(np.abs(S - 0.0))

    idx_h_max = np.argmin(np.abs(h - h[idx_S_max]))

    KaKf = (1 + r_section / (r_section + h))**2

    m_arr = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    Cn_Beta_arr = np.array([0.96667, 0.819047, 0.66667, 0.523809, 0.4142857, 0.3285714286])
    coefficients_CnBeta = np.polyfit(m_arr, Cn_Beta_arr, deg=5)
    
    m_arr_pol = np.linspace(m_arr[0], m_arr[-1], 1000)
    Cn_Beta_pol = np.polyval(coefficients_CnBeta, m)

    Cn_isolated = (4/np.sqrt(3)) * Cn_Beta_pol

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    gs = GridSpec(3, 2, figure=fig)

    axs = []
    axs.append(fig.add_subplot(gs[0, 0]))  # Graphique de h
    axs.append(fig.add_subplot(gs[0, 1]))  # Graphique de S et S/S_ref
    axs.append(fig.add_subplot(gs[1, 0]))  # Graphique de KaKf
    axs.append(fig.add_subplot(gs[1, 1]))  # Graphique de Cn_Beta
    axs.append(fig.add_subplot(gs[2, :]))  # Graphique de Cn_alpha

    # Graphique de h
    axs[0].plot(m, h, c='navy')
    axs[0].set_xlabel('m', fontsize=12)
    axs[0].set_ylabel('h', fontsize=12)
    axs[0].grid(True)

    # Graphique de Cn_Beta
    axs[1].plot(m, Cn_Beta_pol, c='navy')
    axs[1].plot(m, Cn_isolated, c='darkorange')
    axs[1].set_xlabel('m', fontsize=12)
    axs[1].set_ylabel(r'$\frac{\beta}{4}Cn_{\alpha}$', fontsize=12)
    axs[1].grid(True)

    # Graphique de S et S/S_ref
    axs[2].plot(h, S, c='navy', label='S')
    axs[2].plot(h, S / S_ref, c='darkorange', label='S/S_ref')

    axs[2].vlines(x=h[idx_S_0], ymin=(S/S_ref).min(), ymax=(S/S_ref).max(), color='red', alpha=0.75, linestyle='-.')
    axs[2].scatter(h[idx_S_0], S[idx_S_0], marker='D', s=100, edgecolor='red', facecolor='none', label=f'S = 0 -> h = {h[idx_S_0]:.5f} m')
    axs[2].vlines(x=h[idx_S_max], ymin=(S/S_ref).min(), ymax=(S/S_ref).max(), color='magenta', alpha=0.75, linestyle='-.')
    axs[2].scatter(h[idx_S_max], S[idx_S_max], marker='*', s=200, edgecolor='magenta', facecolor='none', label=f'S = {S[idx_S_max]:.5f} m² -> h = {h[idx_S_max]:.5f} m')

    axs[2].set_xlabel('h', fontsize=12)
    axs[2].set_ylabel('Surface', fontsize=12)
    axs[2].legend(fontsize=10)
    axs[2].grid(True)

    # Graphique de Ka + Kf
    axs[3].plot(h, KaKf, c='navy')
    axs[3].set_xlabel('h', fontsize=12)
    axs[3].set_ylabel(r'$K_A + K_F$', fontsize=12)
    axs[3].grid(True)

    # Calcul de Cn_alpha
    Cn_alpha = Cn_isolated * KaKf * (S / S_ref)

    # Graphique de Cn_alpha
    idx = np.argmin(np.abs(Cn_isolated - Cn_section))
    idx_interaction = np.argmin(np.abs(Cn_alpha - Cn_alpha[idx]))
    optimal_height = h[idx_interaction]


    axs[4].plot(h[idx_h_max:], Cn_alpha[idx_h_max:], c='navy')
    axs[4].vlines(x=h[idx_h_max], ymin=Cn_alpha[idx_h_max:].min(), ymax=Cn_alpha[idx_h_max:].max(), color='magenta', alpha=0.75, linestyle='-.')
    axs[4].scatter(h[idx_h_max], Cn_alpha[idx_h_max], marker='*', s=200, edgecolor='magenta', facecolor='none', label=f'Cn_alpha_max = {Cn_alpha[idx_h_max]:.5f} pour h = {h[idx_h_max]:.5f} m')

    axs[4].vlines(x=optimal_height, ymin=Cn_alpha[idx_h_max:].min(), ymax=Cn_alpha[idx] + 1, color='green', alpha=0.75, linestyle='--')
    axs[4].hlines(y=Cn_alpha[idx], xmin=h.min(), xmax=optimal_height + 0.05, color='green', alpha=0.75, linestyle='--')
    axs[4].scatter(optimal_height, Cn_alpha[idx], marker='8', s=200, edgecolor='green', facecolor='none', label=f'Cn_alpha_target = {Cn_alpha[idx]:.5f} pour h = {optimal_height:.5f} m')

    axs[4].set_xlabel('h', fontsize=12)
    axs[4].set_ylabel(r'$Cn_{\alpha}$', fontsize=12)
    axs[4].grid(True)
    axs[4].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    data_dict = {
        'm': m_arr_pol,
        'h': h,
        'Cn_Beta':Cn_Beta_pol,
        'Cn_iso': Cn_isolated,
        'Ka + Kf': KaKf,
        'S/S_ref': S / S_ref,
        'Cn_alpha': Cn_alpha

    }

    return data_dict, optimal_height