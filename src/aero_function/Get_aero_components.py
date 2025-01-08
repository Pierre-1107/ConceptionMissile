import numpy as np
from termcolor import colored
from scipy.optimize import newton


def Get_aero_components(cst_values, Cn_dict, xd_position, diameter, dX, type_cons, CGx_dict, mass_dict, row):

    ## ----- GESTION DES ARGUMENTS ----- ##

            # coefficient de portance 
    Cn_F = Cn_dict['Cn_F']
    Cn_EA = Cn_dict['Cn_EA']

            # x_position / d
    xd_F = xd_position['xd_F']

            # gestion des constantes
    n = cst_values['n']
    g = cst_values['g']
    p = cst_values['p']
    gamma = cst_values['gamma']
    Mach = cst_values['Mach']
    alpha_max = cst_values['alpha']
    S_ref = cst_values['S_ref']

        # ----- centre de gravité ----- #
    if type_cons == 'PISTON':
        xd_position['xd_CG'] = CGx_dict[type_cons][1000:].max() / diameter
        CG_idx = np.argmin(np.abs(CGx_dict[type_cons][1000:] - xd_position['xd_CG']))
        mass = mass_dict['MASS_TOT'][1000:][CG_idx]
    elif type_cons == 'BLADDER':
        xd_position['xd_CG'] = CGx_dict[type_cons][1000:].max() / diameter
        CG_idx = np.argmin(np.abs(CGx_dict[type_cons][1000:] - xd_position['xd_CG']))
        mass = mass_dict['MASS_TOT'][1000:][CG_idx]
    else:
        xd_position['xd_CG'] = 0.0
        CG_idx = np.argmin(np.abs(CGx_dict[type_cons][1000:] - xd_position['xd_CG']))
        mass = mass_dict['MASS_TOT'][1000:][CG_idx]

    ## ----- expression de la constante ----- ##
    num = 2 * n * mass * g
    den = p * gamma * Mach**2 * S_ref
    constant = num/den

        # ----- calcul des x_i/d ----- #
    xd_position['xd_EA'] = (row['L_ogive'] + row['L_equipement'] + row['L_payload'] + dX) / diameter
    # xd_position['xd_W'] = (row['L_ogive'] + row['L_equipement'] + row['L_payload'] + 0.5 * row['L_cruise_res']) / diameter
    xd_position['xd_W'] = (row['L_ogive'] + row['L_equipement'] + row['L_payload'] + 0.5 * row['L_cruise_res'] - 0.75 * diameter ) / diameter
    # xd_position['xd_T'] = (row['L_ogive'] + row['L_payload'] + row['L_equipement'] + row['L_cruise_res'] + row['L_engine_housing'] + row['L_acc_res'] - 0.5 * diameter) / diameter
    xd_position['xd_T'] = (row['L_ogive'] + row['L_payload'] + row['L_equipement'] + row['L_cruise_res'] + row['L_engine_housing'] + row['L_acc_res'] - 0.5 * diameter - 0.25 *diameter) / diameter
    xd_position['Delta_d'] = 0.75
        # ----- calcul des Cn_i ----- #
    Cn_dict['Cn_EA'] = 5.95

    def Cn_W_equation(Cn_w):
        Cn_1 = Cn_F + Cn_EA + Cn_w
        xd_f1 = (Cn_F * xd_F + Cn_EA * xd_position['xd_EA'] + xd_position['xd_W'] * Cn_w) / Cn_1
        left = alpha_max / np.cos(alpha_max)
        right = constant * (xd_position['xd_T'] - xd_position['xd_CG']) / (Cn_1 * (xd_position['xd_T'] - xd_f1))
        return left - right

    initial_guess_Cn_W = 3.0
    Cn_dict['Cn_W'] = newton(Cn_W_equation, initial_guess_Cn_W)

    def Cn_T_equation(Cn_T):
        num_alpha = Cn_F * xd_F + Cn_EA * xd_position['xd_EA'] + Cn_dict['Cn_W'] * xd_position['xd_W'] + Cn_T * xd_position['xd_T'] 
        den_alpha = Cn_F + Cn_EA + Cn_dict['Cn_W'] + Cn_T
        left = num_alpha / den_alpha
        right = xd_position['Delta_d'] + xd_position['xd_CG']
        return left - right
    
    initial_guess_Cn_T = 10
    Cn_dict['Cn_T'] = newton(Cn_T_equation, initial_guess_Cn_T)

    ## ----- DIMENSIONNEMENT SANS AILES ET GOUVERNES ----- ##
    print(f"{colored('Information sans ailes et gouvernes : ', 'yellow')}")
    Cn_alpha_sWT = Cn_F + Cn_EA
    xd_f1_sWT = (Cn_F * xd_F + Cn_EA * xd_position['xd_EA']) / Cn_alpha_sWT
    ratio_sWT = (xd_position['xd_T'] - xd_position['xd_CG'])/(xd_position['xd_T'] - xd_f1_sWT)
    alpha_sWT = np.arctan(constant * (1/Cn_alpha_sWT) * ratio_sWT)

    print(f"    - {colored('Position du foyer aéro :', 'blue')} {xd_f1_sWT}, soit --> xf_1 = {xd_f1_sWT * diameter} m")
    print(f"    - {colored('Coefficient de portance : ', 'blue')} {Cn_alpha_sWT}")
    print(f"    - {colored('Angle d attaque max : ', 'blue')} : {np.rad2deg(alpha_sWT)} °\n")

    ## ----- DIMENSIONNEMENT SANS GOUVERNES ----- ##
    print(f"{colored('Information sans gouvernes : ', 'yellow')}")
    Cn_alpha_sT = Cn_F + Cn_EA + Cn_dict['Cn_W']
    xd_f1_sT = (Cn_F * xd_F + Cn_EA * xd_position['xd_EA'] + Cn_dict['Cn_W'] * xd_position['xd_W']) / Cn_alpha_sT
    ratio_sT = (xd_position['xd_T'] - xd_position['xd_CG'])/(xd_position['xd_T'] - xd_f1_sT)
    alpha_sT = np.arctan(constant * (1/Cn_alpha_sT) * ratio_sT)

    print(f"    - {colored('Position du foyer aéro :', 'blue')} {xd_f1_sT}, soit --> xf_1 = {xd_f1_sT * diameter} m")
    print(f"    - {colored('Coefficient de portance : ', 'blue')} {Cn_alpha_sT}")
    print(f"    - {colored('Angle d attaque max : ', 'blue')} : {np.rad2deg(alpha_sT)} °")

    ## ----- DIMENSIONNEMENT COMPLET ----- ##
    
    return Cn_dict, xd_position