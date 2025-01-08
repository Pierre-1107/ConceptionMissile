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
from aero_function.Get_Data_section import Get_Data_section

## -------------------------------- ##

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

## DONNÉES
t_acc = 4.85655564067031
t_cruise = 143.82010925319975
t_mission = t_acc + t_cruise

    # thermophysique
gamma = 1.4
P_amb = 101325
T_amb = 288
r_air = 8.314 / (28.966*1e-3)
g0 = 9.80665
Mach_cruise = 2.0
Mach_acc = 0.6
V_son = np.sqrt(gamma * r_air * T_amb)
Delta_Vr = (Mach_cruise - Mach_acc) * V_son

    # Impulsion spécifique
Isp_acc = 240

    # masse volumique
rho_a = 1800
rho_c = 1000
rho_payload = 3000
rho_equipement = 1500

    # masses
m_ogive = 25
m_equipement = 55
m_payload = 200
m_engine_h = 10
m_nozzle = 12
m_tails = 16

    # coefficient
c_a = 0.7
i_a = 0.28
c_c = 0.9
i_c = 0.22

    # diamètre vecteur
dt = 0.001
d_missile = np.arange(0.2, 0.6 + dt, dt)
shape = d_missile.shape[0]

keys_dict_main, cruise_data_dict, Cx_acc_arr, Mach_acc_arr, isp_value, Cd_c, densite_value = generate_Cx_Isp(Mach_cruise=Mach_cruise, shape=d_missile.shape[0], 
                                                                                                             img_path=img_path)

time = {
    't_acc': t_acc,
    't_cruise': t_cruise,
}

data_mission = {
    'gamma': gamma,
    'P_amb': P_amb,
    'Mach_cruise': Mach_cruise,
    'g0': g0,
    'Cd_c': Cd_c,
    'Delta_Vr': Delta_Vr,
    'Isp_acc': Isp_acc,
    'mass': {
        'm_ogive': m_ogive,
        'm_equipement': m_equipement,
        'm_tails': m_tails,
        'm_payload': m_payload,
        'm_engine_h': m_engine_h,
        'm_nozzle': m_nozzle
    },
    'rho': {
        'rho_payload': rho_payload,
        'rho_equipement': rho_equipement,
        'rho_c': rho_c,
        'rho_a': rho_a
    }
}

c = {
    'c_a': c_a,
    'c_c': c_c
}

i = {
    'i_a': i_a,
    'i_c': i_c
}

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

oxydizer = "Liquide dense"
idx_chosen_oxydizer = np.where(keys_dict_main == oxydizer)[0]

## ----- CHARGEMENT DU FICHIER ----- ##
results_path = os.path.join(resultats_path, f"{oxydizer}.csv")
results_df = pd.read_csv(results_path, sep="\t")

## ----- CARCATÉRISTIQUES DU MISSILE ----- ##
diametre = 0.35
idx_diametre = np.argmin(np.abs(results_df['d_m'] - diametre))
row = results_df.iloc[idx_diametre]

choosen_oxydiser = "Liquide dense"

constraint_graphs(choosen_oxydiser=choosen_oxydiser, keys_dict_main=keys_dict_main, 
                  length_tensor=length_tensor, mass_tensor=mass_tensor, 
                  d_missile=d_missile, isp_value=isp_value, 
                  cruise_data_dict=cruise_data_dict, img_path=img_path
                  )

x_ogive, y_ogive, x_CG_ogive = generate_ogive_shape(row=row, diametre=diametre, img_path=img_path)

L_missile = row[1]
print(f"Longueur du missile : {colored(L_missile, 'blue')} m.")


section_missile = generate_missile_section(row=row, x_ogive=x_ogive, y_ogive=y_ogive, x_CG_ogive=x_CG_ogive)


mc_dot = row['m_cruise_prop'] / t_cruise
ma_dot = row["m_acc_prop"] / t_acc

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

thermophysic = {
    'P_inf': 101325,
    'T_inf': 288.15,
    'gamma': 1.4,
    'r': 0.4,
}

oxydizer_density = 1.0

abacusvalue = {
    'omega': 0.1278,
    'theta': 0.5556,
    'sigma': 1.688
}

airintake_params = {
    'eps': 0.925,
    'eta': 0.85,
}

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
print(f"\nTaille de la souris en sortie en entrée d'entrée d'air : {colored(delta_X, 'magenta')} m.")

dX = 100.998 *1e-3

Cn_dict = {
    'Cn_F': 2.95,
    'Cn_EA': 5.95
}

xd_position = {
    'xd_F': 2.5,
}

cst_values = {
    'n': 10,
    'g': 9.80665,
    'p': 101325,
    'gamma': 1.4,
    'Mach': Mach_cruise,
    'alpha': np.deg2rad(6),
    'S_ref' : 0.25 * np.pi * diametre**2
}


Cn_dict, xd_position = Get_aero_components(cst_values=cst_values, Cn_dict=Cn_dict, xd_position=xd_position, diameter=diametre, dX=dX, type_cons='PISTON', CGx_dict=CGx_dict, mass_dict=mass_dict, row=row)

print('')
for key, value in xd_position.items():
    print(f"{key} = {value}")

print('')
for key, value in Cn_dict.items():
    print(f"{key} = {value}")

## ----- Données des ailes ----- ##

dimension_Wings = {
    'c_section': 3*diametre,
    'diametre': diametre,
    'r_section': diametre/2 + h_AirIntakes,
    'h_AirIntakes': h_AirIntakes,
    'h_wing': None
}

angle = {
    'lambda': np.deg2rad(30),
    'phi': np.deg2rad(5)
}

data_Wings = Get_Data_section(dimension=dimension_Wings, mach=Mach_cruise, section='W', angle=angle)

m_values = [0.5, 1.0, 1.5, 2.0, 2.5]
indices = {f"idx_{int(m*10)}": np.argmin(np.abs(data_Wings['m'] - m)) for m in m_values}

data_WF = {col: [data_Wings[col][idx] for idx in indices.values()] for col in data_Wings.keys()}

data_DF_WF = pd.DataFrame(data_WF)
print(data_DF_WF)

## ----- CHOIX DE LA HAUTEUR ----- ##
Cn_1w = Cn_dict['Cn_W'] / 4
print(f"\n{colored('Coefficient de portance pour une aile : ', 'yellow')} {Cn_1w}")

idx_wing = np.argmin(np.abs(data_Wings['Cn_iso'] - Cn_1w))
m_wing = data_Wings['m'][idx_wing]
h_wing = data_Wings['h'][idx_wing]

print(f"{colored('Données relatives à une aile :', 'red')}")
print(f"    - {colored('m : ', 'blue')} {m_wing}")
print(f"    - {colored('h_wing : ', 'blue')} {h_wing}")

## ----- Données des gouvernes ----- ##

dimension_Tails = {
    'c_section': diametre,
    'diametre': diametre,
    'r_section': 0.5 * (diametre + h_AirIntakes),
    'h_AirIntakes': h_AirIntakes,
    'h_wing': h_wing
}

angle = {
    'lambda': np.deg2rad(30),
    'phi': np.deg2rad(5)
}

data_Tails = Get_Data_section(dimension=dimension_Tails, mach=Mach_cruise, section='T', angle=angle)

m_values = [0.5, 1.0, 1.5, 2.0, 2.5]
indices = {f"idx_{int(m*10)}": np.argmin(np.abs(data_Tails['m'] - m)) for m in m_values}

data_TF = {col: [data_Tails[col][idx] for idx in indices.values()] for col in data_Tails.keys()}

data_DF_TF = pd.DataFrame(data_TF)
print(data_DF_TF)

## ----- CHOIX DE LA HAUTEUR ----- ##
Cn_1T = Cn_dict['Cn_T'] / 4
print(f"\n{colored('Coefficient de portance pour une gouverne : ', 'yellow')} {Cn_1T}")

idx_tail = np.argmin(np.abs(data_Tails['Cn_iso'] - Cn_1T))
m_tail = data_Wings['m'][idx_tail]
h_tail = data_Wings['h'][idx_tail]

print(f"{colored('Données relatives à une gouverne :', 'red')}")
print(f"    - {colored('m : ', 'blue')} {m_tail}")
print(f"    - {colored('h_tail : ', 'blue')} {h_tail}")