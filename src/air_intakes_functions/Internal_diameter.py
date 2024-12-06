import numpy as np
from termcolor import colored

def Get_internal_diameter(external_diameter, airintake_params, abacusvalue):

    ## ----- HYPTOHÈSE À VÉRIFIER ----- ##
    max_mach = 0.3
    diameter_ratio = 1.2

    ## ----- GESTION DES ARGUMENTS ----- ##

    eta = airintake_params['eta']
    sigma = abacusvalue['sigma']

    ## ----- INTERPOLATION VECTOR ----- ##
    sigma_array = np.array([3.9103, 3.6727, 3.4635, 3.2779, 3.1123, 2.9635, 2.8293, 2.7076, 2.5968, 2.4956, 2.4027, 2.3173, 2.2385, 2.1656, 2.0979, 2.0351, 1.9765])
    mach_array = np.array([0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31])

    internal_sigma = sigma * eta * diameter_ratio**2

    internal_mach = np.interp(internal_sigma, sigma_array[::-1], mach_array[::-1])
    internal_diameter = diameter_ratio * external_diameter

    if internal_mach <= max_mach:

        print(f"{colored('L hypothèse est vérifiée, nous avons donc :', 'green')}")
        print(f"    - diamètre externe : d_1 = {colored(external_diameter, 'yellow')} m.")
        print(f"    - diamètre interne : d_2 = {colored(internal_diameter, 'yellow')} m.\n")

        return internal_diameter, internal_mach
    
    else:

        print(f"{colored('L hypothèse n est pas vérifiée !', 'red')}")

        return internal_diameter, internal_mach