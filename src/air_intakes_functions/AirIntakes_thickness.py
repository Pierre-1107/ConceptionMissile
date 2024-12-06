import numpy as np

def Get_AI_thickness(thermophysic, airintake_params, Upstream_Stag, mach, rupture_constraint, d_missile):

    ## ----- GESTION DES ARGUMENTS ----- ##

    P_inf = thermophysic['P_inf']
    gamma = thermophysic['gamma']
    P_stag_0 = Upstream_Stag['P_stag']
    eta_02 = airintake_params['eta']
    
    ## ----- CALCUL DE L'ÉPAISSEUR ----- ##

    P_stag_2 = eta_02 * P_stag_0
    
    def inverse_stag(mach, gamma, pressure_stag):

        return pressure_stag * (1 + 0.5 * (gamma - 1) * mach**2)**(-gamma / (gamma -1))
    
    P_in_AI = inverse_stag(mach=mach, gamma=gamma, pressure_stag=P_stag_2)
    
    real_thickness = np.abs(P_inf - P_in_AI) * d_missile / (2 * rupture_constraint)

    if real_thickness <= 1e-3:
        thickness = 1e-3
    else:
        thickness = real_thickness

    return thickness, real_thickness