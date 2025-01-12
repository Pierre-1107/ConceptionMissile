import numpy as np

def Get_Steering_Angle(cst_values, CGx_dict, mass_dict, LiftCoeff, type_cons, xd_position):

    ## ----- Gestion des arguments ----- ##

            # gestion des coefficients de portance
    Cn_alpha_1 = LiftCoeff['Cn_F'] + 0.25 * LiftCoeff['Cn_EA'] + 0.25 * LiftCoeff['Cn_WF']
    Cn_alpha_2 = 0.25 * LiftCoeff['Cn_TF']
    Cn_alpha = Cn_alpha_1 + Cn_alpha_2

            # gestion des constantes
    n = cst_values['n']
    g = cst_values['g']
    p = cst_values['p']
    gamma = cst_values['gamma']
    Mach = cst_values['Mach']
    alpha_max = cst_values['alpha']
    S_ref = cst_values['S_ref']

            # coeff de braquage
    Cn_delta = (2/3) * Cn_alpha

            # ----- centre de gravité ----- #
    if type_cons == 'PISTON':
        CG_idx = np.argmin(np.abs(CGx_dict[type_cons][1000:] - xd_position['xd_CG']))
        mass = mass_dict['MASS_TOT'][1000:][CG_idx]
    elif type_cons == 'BLADDER':
        CG_idx = np.argmin(np.abs(CGx_dict[type_cons][1000:] - xd_position['xd_CG']))
        mass = mass_dict['MASS_TOT'][1000:][CG_idx]
    else:
        CG_idx = np.argmin(np.abs(CGx_dict[type_cons][1000:] - xd_position['xd_CG']))
        mass = mass_dict['MASS_TOT'][1000:][CG_idx]

    num = n * mass * g * np.cos(alpha_max)
    den = 0.5 * p * gamma * Mach**2 * S_ref
    constant = num / den

    delta_max = constant * (1/Cn_delta) - ((Cn_alpha_1 + Cn_alpha_2) * alpha_max) / Cn_delta

    return delta_max