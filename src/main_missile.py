import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import os
import pandas as pd
from typing import Tuple
from termcolor import colored
from pathlib import Path

from scipy.interpolate import interp1d
from scipy.integrate import trapz, quad

## ----- IMPORT DES FONCTIONS ----- ##

from simulation_data import data_mission, c, i, time, thermophysic, abacusvalue, airintake_params, Cn_dict, xd_position, cst_values, angle

from dimensioning_functions.generate_Cx_Isp import generate_Cx_Isp
from dimensioning_functions.constraint_algo import constraints_algo
from dimensioning_functions.generate_csv_txt import generate_csv_txt
from dimensioning_functions.constraint_graphs import constraint_graphs

from design_functions.generate_missile_sections import generate_missile_section
from design_functions.generate_ogive_shape import generate_ogive_shape
from design_functions.generate_CG_missile import generate_CG_missile

from air_intakes_functions.External_diameter import Get_external_diameter
from air_intakes_functions.Internal_diameter import Get_internal_diameter
from air_intakes_functions.AirIntakes_thickness import Get_AI_thickness

from aero_function.Get_aero_components import Get_aero_components
from aero_function.Get_Steering_Angle import Get_Steering_Angle
from aero_function.Get_section_component import Get_section_component
from aero_function.Get_drag_data import Get_drag_data

## ================================= ##
## ----- CRÉATION DES DOSSIERS ----- ##
## ================================= ##

docs_path = Path.cwd().parent / "Docs"
img_path = docs_path / "img"
resultats_path = docs_path / "resultats"

# Création des dossiers si nécessaire
for path in [docs_path, img_path, resultats_path]:
    if not path.exists():
        path.mkdir(parents=True)
        print(f"Le dossier {colored(path.name, 'green')} a été créé.")
    else:
        print(f"Le dossier {colored(path.name, 'yellow')} existe déjà.")

## ==================================== ##
## ----- ALGORITHME DE CONTRAINTE ----- ##
## ==================================== ##

dt = 0.001
d_missile = np.arange(0.2, 0.6 + dt, dt)
shape = d_missile.shape[0]

keys_dict_main, cruise_data_dict, Cx_acc_arr, Mach_acc_arr, isp_value, Cd_c, densite_value = generate_Cx_Isp(Mach_cruise=data_mission['Mach_cruise'], shape=d_missile.shape[0], 
                                                                                                             img_path=img_path)

data_mission['Cd_c'] = Cd_c

graph_component = {
    'Mach_acc_arr': Mach_acc_arr,
    'Cx_acc_arr': Cx_acc_arr
}

iterate = {
    'keys_dict_main': keys_dict_main,
    'cruise_data_dict': cruise_data_dict
}


mass_tensor, length_tensor, d_missile = constraints_algo(time=time, data_mission=data_mission, c=c, i=i, graph_component=graph_component, iterate=iterate)

generate_csv_txt(cruise_data_dict=cruise_data_dict, d_missile=d_missile, 
                 data_mission=data_mission, length_tensor=length_tensor, 
                 mass_tensor=mass_tensor, isp_value=isp_value, 
                 densite_value=densite_value, resultats_path=resultats_path, c=c)

    ## ----- OXYDANT ----- ## 
# "Kerosene", "Liquide dense", "Bore", "Hydrocarbure"

choosen_oxydiser = "Liquide dense"
idx_chosen_oxydizer = np.where(keys_dict_main == choosen_oxydiser)[0]

    ## ----- CHARGEMENT DU FICHIER ----- ##
results_path = os.path.join(resultats_path, f"{choosen_oxydiser}.csv")
results_df = pd.read_csv(results_path, sep="\t")

#   # ----- CARCATÉRISTIQUES DU MISSILE ----- ##
diametre = 0.35
idx_diametre = np.argmin(np.abs(results_df['d_m'] - diametre))
row = results_df.iloc[idx_diametre]

constraint_graphs(choosen_oxydiser=choosen_oxydiser, keys_dict_main=keys_dict_main, 
                  length_tensor=length_tensor, mass_tensor=mass_tensor, 
                  d_missile=d_missile, isp_value=isp_value, 
                  cruise_data_dict=cruise_data_dict, img_path=img_path
                  )

## ================================ ##
## ----- ALGORITHME DE DESIGN ----- ##
## ================================ ##

x_ogive, y_ogive, x_CG_ogive, coeff_ogive = generate_ogive_shape(row=row, diametre=diametre, img_path=img_path)

L_missile = row[1]
print(f"Longueur du missile : {colored(L_missile, 'blue')} m.")


section_missile = generate_missile_section(row=row, x_ogive=x_ogive, y_ogive=y_ogive, x_CG_ogive=x_CG_ogive)

mc_dot = row['m_cruise_prop'] / time['t_cruise']
ma_dot = row["m_acc_prop"] / time['t_acc']

m_dot = {
    'ma_dot': ma_dot,
    'mc_dot': mc_dot
}

CGx_dict, mass_dict, t_tot_array = generate_CG_missile(time=time, m_dot=m_dot, 
                                                       section_missile=section_missile, 
                                                       row=row, diametre=diametre, img_path=img_path)

print(f"\n{colored('CG --> Transition acc - cruise pour vessie : ', 'blue')}")
print(CGx_dict['BLADDER'][999])
print(CGx_dict['BLADDER'][1000])
print(CGx_dict['BLADDER'][1001])

print(f"\n{colored('CG --> Transition acc - cruise pour piston : ', 'blue')}")
print(CGx_dict['PISTON'][999])
print(CGx_dict['PISTON'][1000])
print(CGx_dict['PISTON'][1001])

print(f"\nDétails des données :\n{row}\n")

## =============================================== ##
## ----- ALGORITHME DESIGN DES ENTRÉES D'AIR ----- ##
## =============================================== ##

oxydizer_density = 1.0
sigma_rupture = 700 * 1e6

external_diameter, Upstream_Stag = Get_external_diameter(thermophysic=thermophysic, oxydizer_density=oxydizer_density,
                                                         abacusvalue=abacusvalue, airintake_params=airintake_params,
                                                         cruise_mass_flow=mc_dot)

print(f"Diamètre externe - entrée d'air : {colored(external_diameter, 'blue')} m.\n")

internal_diameter, internal_mach = Get_internal_diameter(external_diameter=external_diameter, airintake_params=airintake_params, 
                                          abacusvalue=abacusvalue)

print(f"Diamètre interne - entrée d'air : {colored(internal_diameter, 'blue')} m.")
print(f"Nombre de mach interne : {colored(internal_mach, 'blue')}.")

thickness, real_thickness = Get_AI_thickness(thermophysic=thermophysic, airintake_params=airintake_params,
                                             Upstream_Stag=Upstream_Stag, mach=internal_mach,
                                             rupture_constraint=sigma_rupture, d_missile=diametre)

print(f"\nÉpaisseur de la couche d'acier : {colored(thickness, 'blue')} m.")
print(f"Épaisseur réelle de la couche d'acier : {colored(real_thickness, 'blue')} m.")

boundary_layer_trap = diametre / 15
h_AirIntakes = internal_diameter + 2 * thickness + boundary_layer_trap
print(f"\nDimensions des entrées d'air : \n     - Longueur : {colored(row['L_AirIntakes'], 'blue')} m\n     - Hauteur : {colored(h_AirIntakes, 'blue')} m")

    ## ----- DONNÉES ----- ##
# provenant ABAQUES REPORT 1135 page 49
mach_secure = 2.2
theta = np.deg2rad(25)
sigma = np.deg2rad(40)

def Get_x_value(diameter, sigma):

    return diameter / (2 * np.tan(sigma))

delta_X = Get_x_value(diameter=external_diameter, sigma=sigma)
print(f"\nTaille de la souris en sortie en entrée d'entrée d'air : {colored(delta_X, 'magenta')} m.\n")

## ========================================================= ##
## ----- ALGORITHME EXPRESSION DES DONNÉES DE PORTANCE ----- ##
## ========================================================= ##

cst_values['S_ref'] = 0.25 * np.pi * diametre**2

Cn_dict, xd_position = Get_aero_components(cst_values=cst_values, Cn_dict=Cn_dict, xd_position=xd_position, diameter=diametre, dX=cst_values['dX'], type_cons='PISTON', CGx_dict=CGx_dict, mass_dict=mass_dict, row=row)

print(f"\n{colored('Position relative au diamètre :', 'yellow')}")
for key, value in xd_position.items():
    print(f"{key} = {value}")
print(f"\n{colored('Coefficient de portance :', 'yellow')}")
for key, value in Cn_dict.items():
    print(f"{key} = {value}")
print('')

m_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    ## ----- DIMENSIONNEMENT DES SURFACES PORTANTES ----- ##

wing_data, wing_height = Get_section_component(corde=3*diametre, r_section=0.5 *diametre + h_AirIntakes, S_ref=0.25*np.pi*diametre**2, title="Données pour une aile", Cn_section=0.25 * Cn_dict['Cn_W'])
tail_data, tail_height = Get_section_component(corde=diametre, r_section=0.5*(diametre + h_AirIntakes), S_ref=0.25*np.pi*diametre**2, title="Données pour une gouverne", Cn_section=0.25 * Cn_dict['Cn_T'])

## ========================================================= ##
## ----- ALGORITHME EXPRESSION DES DONNÉES DE TRAÎNÉES ----- ##
## ========================================================= ##

diametre_dict = {
    'diametre': diametre,
    'internal_diameter': internal_diameter,
    'external_diameter': external_diameter,
    'thickness': thickness,
    'h_AirIntakes': h_AirIntakes,
    'S_ref': 0.25 * np.pi * diametre**2
}

target_dict = {
    'ogive': np.sqrt(3) * (diametre/row['L_ogive']),
    'trap': 1 / 15,
    'phi': 15,
    'ratio': ((internal_diameter + 2*thickness) / external_diameter)**2,
    'mach': data_mission['Mach_cruise']
}

airfoil_data = {
    'aile': {
        'ratio_ec': 3/100,
        'K': 6,
        'S': (wing_height * (2 * 3 * diametre) - wing_height * np.tan(np.deg2rad(30))) / (0.25 * np.pi * diametre**2),
        'height': wing_height,
        'corde': 3 * diametre
    },

    'gouverne': {
        'ratio_ec': 6/100,
        'K': 4,
        'S': (tail_height * (2 * 3 * diametre) - tail_height * np.tan(np.deg2rad(30))) / (0.25 * np.pi * diametre**2),
        'height': tail_height,
        'corde': diametre
    },

    'ogive': {
        'coeff': coeff_ogive,
        'length': row['L_ogive']
    }
}

drag_dict = Get_drag_data(target_dict, diametre_dict, row, airfoil_data)
Cx_tot = np.sum([Cx_val for Cx_val in drag_dict.values()]) * 1.1
print(f'{colored("Coefficient de traînée totale : ", "yellow")} {Cx_tot}')