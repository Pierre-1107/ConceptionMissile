import numpy as np

gamma = 1.4
T_amb = 288
r_air = 8.314 / (28.966*1e-3)
Mach_cruise = 2.0
Mach_acc = 0.6

V_son = np.sqrt(gamma * r_air * T_amb)
Delta_Vr = (Mach_cruise - Mach_acc) * V_son

## ----- DONNÉES PRINCIPAL DE LA SIMULATION ----- ##

data_mission = {
    'gamma': 1.4,
    'P_amb': 101325,
    'Mach_cruise': 2,
    'g0': 9.80665,
    'Delta_Vr': Delta_Vr,
    'Isp_acc': 240,
    'mass': {
        'm_ogive': 25,
        'm_equipement': 55,
        'm_tails': 16,
        'm_payload': 200,
        'm_engine_h': 10,
        'm_nozzle': 12
    },
    'rho': {
        'rho_payload': 3000,
        'rho_equipement': 1500,
        'rho_c': 1000,
        'rho_a': 1800
    }
}

## ----- TEMPS RELATIF À LA MISSION ----- ##

time = {
    't_acc': 4.85655564067031,
    't_cruise': 143.82010925319975,
}

## ----- COEFFICIENT DE REMPLISAGE ----- ##

c = {
    'c_a': 0.7,
    'c_c': 0.9
}

## ----- COEFFICIENT CONSTRUCTIF ----- ##

i = {
    'i_a': 0.28,
    'i_c': 0.22
}

## ----- DIMENSIONNEMENT DES ENTRÉES D'AIR ----- ##

thermophysic = {
    'P_inf': 101325,
    'T_inf': 288.15,
    'gamma': 1.4,
    'r': 0.4,
}

abacusvalue = {
    'omega': 0.1278,
    'theta': 0.5556,
    'sigma': 1.688
}

airintake_params = {
    'eps': 0.925,
    'eta': 0.85,
}

## ----- DONNÉES DE PORTANCE ----- ##

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
    'Mach': data_mission['Mach_cruise'],
    'alpha': np.deg2rad(6),
    'dX' :100.998 *1e-3
}

angle = {
    'lambda': np.deg2rad(30),
    'phi': np.deg2rad(5)
}