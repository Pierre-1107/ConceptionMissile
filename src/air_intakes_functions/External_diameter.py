import numpy as np
from typing import Tuple
from termcolor import colored

def Get_external_diameter(thermophysic, oxydizer_density, abacusvalue, airintake_params, cruise_mass_flow):

    ## ----- GESTION DES ARGUMENTS ----- ##

    P_inf = thermophysic['P_inf']
    T_inf = thermophysic['T_inf']
    r = thermophysic['r']

    omega = abacusvalue['omega']
    theta = abacusvalue['theta']
    sigma = abacusvalue['sigma']

    eps = airintake_params['eps']

    ## ----- EXPRESSION DES DIMENSIONS DES ENTRÉES D'AIR ----- ##

    f_s_fnc = lambda density: ((density - 0.8)/(1.6 - 0.8)) * (0.074 - 0.068) + 0.068

    def stagnation_value(omega, theta, pressure, temperature) -> Tuple[float, float]:

        P_stag = pressure / omega
        T_stag = temperature / theta

        return P_stag, T_stag

    def m_air_dot_fnc(cruise_mass_flow, f_stoch, r) -> float:

        m_air_dot = cruise_mass_flow / (r * f_stoch)

        ## CALCUL À LA MAIN m_air_dot environ 94 kg/s
        # m_air_dot = 94
        print(m_air_dot)
        return m_air_dot
    
    def section_c0_fnc(P_i0, T_i0, air_mass_flow) -> float:

        return np.divide(air_mass_flow * np.sqrt(T_i0), 0.04042 * P_i0)
    
    def external_intakes_diameter(section_c0, sigma, eps) -> float:

        section_0 = sigma * section_c0
        section_1_prime = section_0 / (4 * eps)

        return np.sqrt(4 * section_1_prime/np.pi)
    
    P_stag, T_stag = stagnation_value(omega=omega, theta=theta, pressure=P_inf, temperature=T_inf)
    m_air_dot = m_air_dot_fnc(cruise_mass_flow=cruise_mass_flow, f_stoch=f_s_fnc(density=oxydizer_density), r=r)
    A_c0 = section_c0_fnc(P_i0=P_stag, T_i0=T_stag, air_mass_flow=m_air_dot)
    external_diameter = external_intakes_diameter(section_c0=A_c0, sigma=sigma, eps=eps)

    Upstream_Stag = {
        'P_stag': P_stag,
        'T_stag': T_stag
    }

    return external_diameter, Upstream_Stag