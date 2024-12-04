import numpy as np
import matplotlib.pyplot as plt
import os

from termcolor import colored


def generate_CG_missile(time, c, m_dot, section_missile, row, diametre, img_path):
    """
    Calcule et trace l'évolution du centre de gravité (CG), de la masse du missile, et génère une image de la configuration du missile.

    Cette fonction calcule le CG du missile en fonction du temps en utilisant deux approches (Bladder et Piston),
    et génère des graphiques pour l'évolution du CG et de la masse. Elle trace également la géométrie complète du missile.

    Args:
        time (dict): Dictionnaire contenant les durées des phases du vol.
                    Clés attendues : 't_acc' (durée de l'accélération), 't_cruise' (durée de la croisière).
        c (dict): Dictionnaire contenant les coefficients de compression.
                Clés attendues : 'c_a' (coefficient pour l'accélération), 'c_c' (coefficient pour la croisière).
        m_dot (dict): Dictionnaire contenant les débits massiques.
                    Clés attendues : 'mc_dot' (débit massique pour la croisière), 'ma_dot' (débit massique pour l'accélération).
        section_missile (list): Liste des sections du missile. Chaque élément est un tuple contenant :
                                (nom de la section, coordonnées x, coordonnées y, longueur, masse initiale,
                                CG local de la section, couleur).
        row (dict): Dictionnaire contenant les dimensions spécifiques du missile (longueurs des sections, etc.).
        diametre (float): Diamètre du missile.
        img_path (str): Chemin d'accès où sauvegarder l'image générée.

    Returns:
        tuple:
            - CGx_dict (dict): Dictionnaire des positions du centre de gravité calculées.
                            Clés : 'BLADDER', 'PISTON' (tableaux des CG pour chaque méthode).
            - mass_dict (dict): Dictionnaire contenant les masses calculées.
                                Clés : 'MASS' (masses par section), 'MASS_TOT' (masses totales).
            - t_tot_array (numpy.ndarray): Tableau des instants de temps simulés.
    """

    ## ----- GESTION DES ARGUMENTS ----- ##
    t_acc = time['t_acc']
    t_cruise = time['t_cruise']

    c_a = c['c_a']
    c_c = c['c_c']

    mc_dot = m_dot['mc_dot']
    ma_dot = m_dot['ma_dot']

    ## ----- VECTEUR DE TEMPS ----- ##
    size_arr = 1000
    t_acc_array = np.linspace(0, t_acc, size_arr)
    t_cruise_array = np.linspace(t_acc, t_acc + t_cruise, size_arr)
    t_tot_array = np.concatenate([t_acc_array, t_cruise_array[1:]])

    ## ----- DÉFINITION DES VECTEURS ----- ##
    CGx_bladder_array = np.zeros(shape=t_tot_array.shape[0])
    CGx_piston_array = np.zeros(shape=t_tot_array.shape[0])
    mass_array = np.zeros(shape=(t_tot_array.shape[0], len(section_missile)))

    for t_idx, t in enumerate(t_tot_array):

        x_start = 0.0
        CG_num_bladder, CG_num_piston = 0.0, 0.0

        for idx, (section_name, x_val, y_val, length, mass, section_CG, color) in enumerate(section_missile):

            if section_name == "TUYÈRE":
                mass_array[t_idx, idx] = mass if t <= t_acc else 0.0

            if not "PROPERGOL" in section_name and not section_name in ["ENTRÉE AIR", "AILES", "QUEUE"]:

                new_CG = x_start + section_CG
                mass_array[t_idx, idx] = mass

                CG_num_bladder += new_CG * mass_array[t_idx, idx]
                CG_num_piston += new_CG * mass_array[t_idx, idx]
                
                x_start += length

            if section_name in ["ENTRÉE AIR", "AILES", "QUEUE"]:

                if section_name == "ENTRÉE AIR":
                    mass_array[t_idx, idx] = mass
                    position_AI = x_start - row['L_nozzle'] - section_CG

                    CG_num_bladder += position_AI * mass_array[t_idx, idx]
                    CG_num_piston += position_AI * mass_array[t_idx, idx]

                elif section_name == "AILES":
                    mass_array[t_idx, idx] = mass
                    position_W = x_start - row['L_nozzle'] - row['L_acc_res'] - row['L_engine_housing'] - section_CG

                    CG_num_bladder += position_W * mass_array[t_idx, idx]
                    CG_num_piston += position_W * mass_array[t_idx, idx]

                elif section_name == "QUEUE":
                    mass_array[t_idx, idx] = mass
                    position_T = x_start - row['L_nozzle'] - section_CG

                    CG_num_bladder += position_T * mass_array[t_idx, idx]
                    CG_num_piston += position_T * mass_array[t_idx, idx]

            if "PROPERGOL" in section_name:
                if section_name == "PROPERGOL ACCÉLÉRATION":
                    mass_array[t_idx, idx] = max(0.0, mass - ma_dot * t) if t <= t_acc else 0.0
                    previous_x_start = x_start - row['L_acc_res']

                    Delta_bladder = length * (1/c_a - 1)
                    x_position = previous_x_start + Delta_bladder

                    CG_num_bladder += (x_position + length/2) * mass_array[t_idx, idx]
                    CG_num_piston += (x_position + length/2) * mass_array[t_idx, idx]

                elif section_name == "PROPERGOL CROISIÈRE":

                    mass_array[t_idx, idx] = max(0.0, mass - 0.9 * mc_dot * (t - t_acc)) if t > t_acc else mass
                    previous_x_start = x_start - row['L_cruise_res']

                    ## ----- BLADDER METHOD ----- ##
                    DeltaL_bladder = length * (1/c_c - 1)
                    x_position_bladder = previous_x_start + DeltaL_bladder
                    CG_num_bladder += (x_position_bladder + length / 2) * mass_array[t_idx, idx]

                    ## ----- PISTON METHOD ----- ##
                    L_cp_t = length * (0.9 * (t_cruise - (t - t_acc))/t_cruise + 0.1) if t > t_acc else length
                    DeltaL_piston = row['L_cruise_res'] - L_cp_t
                    x_position_piston = previous_x_start + DeltaL_piston
                        
                    CG_num_piston += (x_position_piston + L_cp_t / 2) * mass_array[t_idx, idx]
                    
        ## ----- EXPRESSION DES CG ----- ##
        CGx_bladder_array[t_idx] = CG_num_bladder / mass_array[t_idx, :].sum()
        CGx_piston_array[t_idx] = CG_num_piston / mass_array[t_idx, :].sum()

    mass_array_2D = np.zeros(shape=t_tot_array.shape[0])
    for idx in range(t_tot_array.shape[0]):
        mass_array_2D[idx] = mass_array[idx, :].sum()

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ## ----- ÉVOLUTION DU CENTRE DE GRAVITÉ ----- ##
    axes[0, 0].set_title('Évolution du centre de gravité', fontsize=16)

    axes[0, 0].plot(t_tot_array, CGx_bladder_array, c="navy", label="BLADDER TANK")
    axes[0, 0].plot(t_tot_array, CGx_piston_array, c="darkorange", label="PISTON TANK")

    axes[0, 0].set_xlabel("Temps [m]")
    axes[0, 0].set_ylabel("Centre de gravité [m]")
    axes[0, 0].grid('on', alpha=0.75, linestyle="-.")
    axes[0, 0].legend(loc="upper right")

    ## ----- ÉVOLUTION DE LA MASSE ----- ##
    axes[0, 1].set_title('Évolution de la masse', fontsize=16)
    axes[0, 1].plot(t_tot_array, mass_array_2D, c='navy', label='Évolution de la masse')
    axes[0, 1].set_xlabel("Temps [m]")
    axes[0, 1].set_ylabel("Masse [kg]")
    axes[0, 1].grid('on', alpha=0.75, linestyle="-.")
    axes[0, 1].legend(loc="upper right")

    fig.delaxes(axes[1, 0])
    fig.delaxes(axes[1, 1])
    merged_axes = fig.add_subplot(2, 1, 2)

    ## ----- MISSILE ----- ##
    merged_axes.set_title('Représentation du missile', fontsize=16)

    x_missile_array = np.linspace(row['L_ogive'], row['L_m'], 2000)
    y_misisle_array = np.full(2000, diametre/2)

    x_start = 0.0
    for section_name, x_val, y_val, length, _, _, color in section_missile:

        if not np.any(np.isnan(x_val)):
            merged_axes.plot(x_val, y_val, color=color)
            merged_axes.plot(x_val, -y_val, color=color)
            merged_axes.fill_between(x=x_val, y1=-y_val, y2=y_val, color=color, hatch="//", alpha=0.5)

        if section_name not in ["PROPERGOL CROISIÈRE", "PROPERGOL ACCÉLÉRATION", "ENTRÉE AIR", "AILES", "QUEUE"]:
            x_position = x_start + length
            merged_axes.vlines(x=x_position, ymin=-diametre/2, ymax=diametre/2, color=color)
            merged_axes.fill_between(x=x_missile_array, y1=-diametre/2, y2=diametre/2, where=(x_missile_array >= x_start) & (x_missile_array <= x_position), color=color, alpha=0.5, hatch="//", label=section_name)
            x_start += length

        if section_name == "ENTRÉE AIR":
            x_start_AI = row['L_ogive'] + row['L_payload'] + row['L_equipement']
            y_min = 0.05 + diametre/2
            y_max = 0.2 + diametre/2

            ## ----- UPPER AIR INTAKE ----- ##
            merged_axes.vlines(x=x_start_AI, ymin=y_min, ymax=y_max, color=color)
            merged_axes.vlines(x=x_start_AI + length, ymin=y_min, ymax=y_max, color=color)
            merged_axes.hlines(y=y_min, xmin=x_start_AI, xmax=x_start_AI + length, color=color)
            merged_axes.hlines(y=y_max, xmin=x_start_AI, xmax=x_start_AI + length, color=color)
            merged_axes.fill_between(x=x_missile_array, y1=y_min, y2=y_max, where=(x_missile_array >= x_start_AI) & (x_missile_array <= x_start_AI + length), color=color, alpha=0.5, hatch="//", label=section_name)

            ## ----- LOWER AIR INTAKE ----- ##
            merged_axes.vlines(x=x_start_AI, ymin=-y_max, ymax=-y_min, color=color)
            merged_axes.vlines(x=x_start_AI + length, ymin=-y_max, ymax=-y_min, color=color)
            merged_axes.hlines(y=-y_min, xmin=x_start_AI, xmax=x_start_AI + length, color=color)
            merged_axes.hlines(y=-y_max, xmin=x_start_AI, xmax=x_start_AI + length, color=color)
            merged_axes.fill_between(x=x_missile_array, y1=-y_max, y2=-y_min, where=(x_missile_array >= x_start_AI) & (x_missile_array <= x_start_AI + length), color=color, alpha=0.5, hatch="//")

        if section_name == "AILES":
            x_start_W = row['L_ogive'] + row['L_payload'] + row['L_equipement']
            y_min = 0.25 + diametre/2
            y_max = 0.4 + diametre/2

            ## ----- UPPER WING ----- ##
            merged_axes.vlines(x=x_start_W, ymin=y_min, ymax=y_max, color=color)
            merged_axes.vlines(x=x_start_W + length, ymin=y_min, ymax=y_max, color=color)
            merged_axes.hlines(y=y_min, xmin=x_start_AI, xmax=x_start_AI + length, color=color)
            merged_axes.hlines(y=y_max, xmin=x_start_AI, xmax=x_start_AI + length, color=color)
            merged_axes.fill_between(x=x_missile_array, y1=y_min, y2=y_max, where=(x_missile_array >= x_start_W) & (x_missile_array <= x_start_W + length), color=color, alpha=0.5, hatch="//", label=section_name)

            ## ----- LOWER WING ----- ##
            merged_axes.vlines(x=x_start_W, ymin=-y_max, ymax=-y_min, color=color)
            merged_axes.vlines(x=x_start_W + length, ymin=-y_max, ymax=-y_min, color=color)
            merged_axes.hlines(y=-y_min, xmin=x_start_AI, xmax=x_start_AI + length, color=color)
            merged_axes.hlines(y=-y_max, xmin=x_start_AI, xmax=x_start_AI + length, color=color)
            merged_axes.fill_between(x=x_missile_array, y1=-y_max, y2=-y_min, where=(x_missile_array >= x_start_W) & (x_missile_array <= x_start_W + length), color=color, alpha=0.5, hatch="//")

        if section_name == "QUEUE":
            x_start_T = row['L_m'] - row['L_nozzle'] - length
            y_min = 0.25 + diametre/2
            y_max = 0.4 + diametre/2

            ## ----- UPPER TAIL ----- ##
            merged_axes.vlines(x=x_start_T, ymin=y_min, ymax=y_max, color=color)
            merged_axes.vlines(x=x_start_T + length, ymin=y_min, ymax=y_max, color=color)
            merged_axes.hlines(y=y_min, xmin=x_start_T, xmax=x_start_T + length, color=color)
            merged_axes.hlines(y=y_max, xmin=x_start_T, xmax=x_start_T + length, color=color)
            merged_axes.fill_between(x=x_missile_array, y1=y_min, y2=y_max, where=(x_missile_array >= x_start_T) & (x_missile_array <= x_start_T + length), color=color, alpha=0.5, hatch="//", label=section_name)

            ## ----- LOWER TAIL ----- ##
            merged_axes.vlines(x=x_start_T, ymin=-y_max, ymax=-y_min, color=color)
            merged_axes.vlines(x=x_start_T + length, ymin=-y_max, ymax=-y_min, color=color)
            merged_axes.hlines(y=-y_min, xmin=x_start_T, xmax=x_start_T + length, color=color)
            merged_axes.hlines(y=-y_max, xmin=x_start_T, xmax=x_start_T + length, color=color)
            merged_axes.fill_between(x=x_missile_array, y1=-y_max, y2=-y_min, where=(x_missile_array >= x_start_T) & (x_missile_array <= x_start_T + length), color=color, alpha=0.5, hatch="//")


    merged_axes.plot(x_missile_array, y_misisle_array, color=section_missile[0][-1])
    merged_axes.plot(x_missile_array, -y_misisle_array, color=section_missile[0][-1])

    merged_axes.legend(loc="upper left", ncol=5)
    merged_axes.grid('on', alpha=0.75, linestyle="-.")
    merged_axes.set_xlabel("Longueur [m]")
    merged_axes.set_xbound([-1, row['L_m'] + 1])
    merged_axes.set_ylabel("Diamètre [m]")
    merged_axes.set_ybound([-1, 1])

    fig_path = os.path.join(img_path, "Missile_CG_Mass.jpg")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', quality=95)
    plt.show()

    CGx_dict = {
        'BLADDER': CGx_bladder_array,
        'PISTON': CGx_piston_array
    }

    mass_dict = {
        'MASS': mass_array,
        'MASS_TOT': mass_array_2D
    }

    print(f"----- MASSE DE PROPERGOL POUR LA PHASE DE CROISIÈRE -----")
    mass_cruise_propergol = mass_array[:, 4]

    mass_start = mass_cruise_propergol[0]
    mass_end = mass_cruise_propergol[-1]

    mass_percentage = (mass_end / mass_start) * 100
    ecart = np.abs(mass_percentage - 10.0)

    print(f"Masse de propergol en fin de croisière : {colored(mass_end, 'green')} kg")
    print(f"Il reste {colored(f'{mass_end:.5f}', 'blue')} kg de propergol soit {colored(f'{mass_percentage:.4f} %', 'blue')} de la masse totale.")
    if ecart < 1e-5:
        print(f"{colored('Condition validé !', 'green')}")
    else:
        print(f"{colored('Condition non validé !', 'red')}")

    return CGx_dict, mass_dict, t_tot_array