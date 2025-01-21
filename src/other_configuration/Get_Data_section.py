import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from termcolor import colored
from matplotlib.gridspec import GridSpec

def Get_Data_section(dimension, mach, section, angle):

    lambda_ = angle['lambda']
    phi = angle['phi']

    c_section = dimension['c_section']
    diametre = dimension['diametre']
    r_section = dimension['r_section']

    def Get_Tails_Surface(c_section, lambda_, phi, h_AirIntakes, h_Tails_arr, h_wing):

        c_T = c_section
        x_h_slope = c_T * np.tan(phi)
        y_h_slope = h_AirIntakes/2 - x_h_slope
        dX_S_NI = (h_wing + y_h_slope) * np.tan(lambda_)

            # ----- surface Non Interractionnée ----- #
        S_NI_prime = (h_wing + y_h_slope) * (2 * c_T - (h_wing + y_h_slope) * np.tan(lambda_))
        S_NI = S_NI_prime + 0.5 * (c_T * x_h_slope)

            # ----- surface Interractionnée ----- #
        dh = h_Tails_arr - (0.5 * h_AirIntakes + h_wing)
        c_T_prime = c_T - dX_S_NI
        S_I = dh * (2 * c_T_prime - dh * np.tan(lambda_))

        K_WT = (0.5 * S_NI + S_I) / (S_NI + S_I)

        return K_WT, S_NI, S_I

    beta = np.sqrt(mach**2 - 1)
    S_ref = 0.25 * np.pi * diametre **2

    m_arr = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    Cn_Beta_arr = np.array([0.96667, 0.819047, 0.66667, 0.523809, 0.4142857])
    height_arr = np.divide(c_section, m_arr * beta)

    m_arr_pol = np.linspace(m_arr[0], m_arr[-1], 1000)

    # graphique [0, 0] --> h = f(m)
    coefficients_height = np.polyfit(m_arr, height_arr, deg=4)
    height_pol = np.polyval(coefficients_height, m_arr_pol)

    # graphique [1, 0] --> (beta /4) * Cn = f(h)
    coefficients_CnBeta = np.polyfit(m_arr, Cn_Beta_arr, deg=5)
    Cn_Beta_pol = np.polyval(coefficients_CnBeta, m_arr_pol)

    # graphique [2, 0] --> Cn = f(h)
    Cn_iso_arr = (4 / beta) * Cn_Beta_arr
    Cn_iso_pol = (4 / beta) * Cn_Beta_pol


    fig, axis = plt.subplots(3, 2, figsize=(15, 12))

    if section == "W":
        # graphique [0, 1] --> K_A + K_F = f(h)
        b_pol = height_pol + r_section
        KaKf_pol = (1 + r_section / b_pol)**2

        # graphique [1, 1] --> S / S_ref = f(h)
        S_A = height_pol * (2 * c_section - height_pol * np.tan(np.deg2rad(lambda_)))
        ratio = S_A / S_ref

        # graphique [2, 1] --> Cn_section/F = f(h)
        Cn_alpha_section_F = Cn_iso_pol * KaKf_pol * ratio

        data_dict = {
            'm': m_arr_pol,
            'h': height_pol,
            'Cn_Beta': Cn_Beta_pol,
            'Cn_iso': Cn_iso_pol,
            'K_A + K_F': KaKf_pol,
            'S/S_ref': ratio,
            'Cn_alpha': Cn_alpha_section_F
        }
        
        axis[1, 1].set_ylabel(r'$\frac{S_w}{S_{red}}$')

    elif section == "T":
        
        # graphique [0, 1] --> K_A + K_F = f(h)
        b_pol = height_pol + r_section
        KaKf_pol = (1 + r_section / b_pol)**2

        K_WT, S_NI, S_I = Get_Tails_Surface(c_section, lambda_, phi, dimension['h_AirIntakes'], height_pol, dimension['h_wing'])
        S_T = (0.5 * S_NI + S_I)/(S_NI + S_I)
        ratio = S_T / S_ref

        Cn_alpha_section_F = Cn_iso_pol * KaKf_pol * ratio * K_WT

        data_dict = {
            'm': m_arr_pol,
            'h': height_pol,
            'Cn_Beta': Cn_Beta_pol,
            'Cn_iso': Cn_iso_pol,
            'K_A + K_F': KaKf_pol,
            'S/S_ref': ratio,
            'K_WT': K_WT,
            'Cn_alpha': Cn_alpha_section_F
        }

        axis[1, 1].set_ylabel(r'$\frac{S_T}{S_{red}}$')

    axis[0, 0].plot(m_arr, height_arr, c='navy', label='Données')
    axis[0, 0].plot(m_arr_pol, height_pol, c='red', linestyle='--', label='Ajustement polynomiale')
    axis[0, 0].legend()
    axis[0, 0].grid('on', alpha=0.75)
    axis[0, 0].set_xlabel('m')
    axis[0, 0].set_ylabel(r'$h_w$')

    axis[1, 0].plot(height_arr, Cn_Beta_arr, c='navy', label='Données')
    axis[1, 0].plot(height_pol, Cn_Beta_pol, c='red', linestyle='--', label='ajustement polynomiale')
    axis[1, 0].legend()
    axis[1, 0].grid('on', alpha=0.75)
    axis[1, 0].set_xlabel(r'$h_w$')
    axis[1, 0].set_ylabel(r'$\frac{\beta}{4} C_{n, \alpha}^{isolé}$')

    axis[2, 0].plot(height_arr, Cn_iso_arr, c='navy', label='Données')
    axis[2, 0].plot(height_pol, Cn_iso_pol, c='red', linestyle='--', label='ajustement polynomiale')
    axis[2, 0].legend()
    axis[2, 0].grid('on', alpha=0.75)
    axis[2, 0].set_xlabel(r'$h_w$')
    axis[2, 0].set_ylabel(r'$C_{n, \alpha}^{isolé}$')

    axis[0, 1].plot(height_pol, KaKf_pol, c='navy', label=r'Coefficient $K_A + K_F$')
    axis[0, 1].set_xlabel(r'$h_w$')
    axis[0, 1].set_ylabel(r'$K_A + K_F$')
    axis[0, 1].legend()
    axis[0, 1].grid('on', alpha=0.75)

    axis[1, 1].plot(height_pol, ratio, c='navy', label='rapport des sections')
    axis[1, 1].set_xlabel(r'$h_w$')
    axis[1, 1].legend()
    axis[1, 1].grid('on', alpha=0.75)

    axis[2, 1].plot(height_pol, Cn_alpha_section_F, c='navy', label=f'Coefficient de portance {section}/F')
    axis[2, 1].set_xlabel(r'$h_w$')
    axis[2, 1].set_ylabel(fr'$C_{{n, \alpha}}^{{{section}/F}}$')
    axis[2, 1].legend()
    axis[2, 1].grid('on', alpha=0.75)

    plt.show()

    return data_dict


## ----- function call ----- ##
# diametre = 0.35
# h_AirIntakes = 0.1746

# dimension_Wings = {
#     'c_section': 3*diametre,
#     'diametre': diametre,
#     'r_section': diametre/2 + h_AirIntakes,
#     'h_AirIntakes': h_AirIntakes,
#     'h_wing': None
# }

# angle = {
#     'lambda': np.deg2rad(30),
#     'phi': np.deg2rad(5)
# }


# data_Wings = Get_Data_section(dimension=dimension_Wings, mach=2.0, section='W', angle=angle)

# m_values = [0.5, 1.0, 1.5, 2.0, 2.5]
# indices = {f"idx_{int(m*10)}": np.argmin(np.abs(data_Wings['m'] - m)) for m in m_values}

# data_WF = {col: [data_Wings[col][idx] for idx in indices.values()] for col in data_Wings.keys()}

# data_DF_WF = pd.DataFrame(data_WF)
# print(data_DF_WF)

# ## ----- CHOIX DE LA HAUTEUR ----- ##
# Cn_1w = Cn_dict['Cn_W'] / 4

# print(f"\n{colored('Coefficient de portance pour une aile : ', 'yellow')} {Cn_1w}")

# idx_wing = np.argmin(np.abs(data_Wings['Cn_iso'] - Cn_1w))
# m_wing = data_Wings['m'][idx_wing]
# h_wing = data_Wings['h'][idx_wing]
# Cn_alpha_WF = data_Wings['Cn_alpha'][idx_wing]

# print(f"{colored('Données relatives à une aile :', 'red')}")
# print(f"    - {colored('m : ', 'blue')} {m_wing}")
# print(f"    - {colored('h_wing : ', 'blue')} {h_wing}")
# print(f"    - {colored('Cn_alpha_WF : ', 'blue')} {Cn_alpha_WF}")

# ## ----- Données des gouvernes ----- ##

# dimension_Tails = {
#     'c_section': diametre,
#     'diametre': diametre,
#     'r_section': 0.5 * (diametre + h_AirIntakes),
#     'h_AirIntakes': h_AirIntakes,
#     'h_wing': h_wing
# }

# angle = {
#     'lambda': np.deg2rad(30),
#     'phi': np.deg2rad(5)
# }

# data_Tails = Get_Data_section(dimension=dimension_Tails, mach=2.0, section='T', angle=angle)

# m_values = [0.5, 1.0, 1.5, 2.0, 2.5]
# indices = {f"idx_{int(m*10)}": np.argmin(np.abs(data_Tails['m'] - m)) for m in m_values}

# data_TF = {col: [data_Tails[col][idx] for idx in indices.values()] for col in data_Tails.keys()}

# data_DF_TF = pd.DataFrame(data_TF)
# print(data_DF_TF)

# ## ----- CHOIX DE LA HAUTEUR ----- ##
# Cn_1T = Cn_dict['Cn_T'] / 4
# print(f"\n{colored('Coefficient de portance pour une gouverne : ', 'yellow')} {Cn_1T}")

# idx_tail = np.argmin(np.abs(data_Tails['Cn_iso'] - Cn_1T))
# m_tail = data_Tails['m'][idx_tail]
# h_tail = diametre / (m_tail * np.sqrt(3))
# Cn_alpha_TF = data_Tails['Cn_alpha'][idx_tail]

# print(f"{colored('Données relatives à une gouverne :', 'red')}")
# print(f"    - {colored('m : ', 'blue')} {m_tail}")
# print(f"    - {colored('h_tail : ', 'blue')} {h_tail}")
# print(f"    - {colored('Cn_alpha_TF : ', 'blue')} {Cn_alpha_TF}")