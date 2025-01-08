import numpy as np
import matplotlib.pyplot as plt
import os

from termcolor import colored

def generate_CG_missile(time, m_dot, section_missile, row, diametre, img_path):

    ## ----- GESTION DES ARGUMENTS ----- ##
    t_acc = time['t_acc']
    t_cruise = time['t_cruise']

    mc_dot = m_dot['mc_dot']
    ma_dot = m_dot['ma_dot']

    ## ----- GESTION DES VECTEURS ----- ##

    t_mission_acc = np.linspace(0, t_acc, 1000)
    t_mission_cruise = np.linspace(t_acc, t_cruise + t_acc, 1000)
    t_mission_array = np.concatenate([t_mission_acc, t_mission_cruise[1:]])

        ## ----- PHASE D'ACCÉLÉRATION ----- ##
    mass_array_tot_acc = np.zeros(shape=(len(section_missile), t_mission_acc.shape[0]))
    CG_position_acc = np.zeros(shape=(len(section_missile), t_mission_acc.shape[0]))
    CG_array_acc = np.zeros(shape=t_mission_acc.shape[0])

        ## ----- PHASE DE CROISIÈRE ----- ##
    mass_array_tot_cruise = np.zeros(shape=(len(section_missile), t_mission_cruise.shape[0]))
    CG_position_cruise_bladder = np.zeros(shape=(len(section_missile), t_mission_cruise.shape[0]))
    CG_position_cruise_piston = np.zeros(shape=(len(section_missile), t_mission_cruise.shape[0]))
    CG_array_cruise_bladder = np.zeros(shape=t_mission_cruise.shape[0])
    CG_array_cruise_piston = np.zeros(shape=t_mission_cruise.shape[0])

    ## ----- SIMULATION POUR LA PHASE D'ACCÉLÉRATION ----- ##

    for t_idx, t_val in enumerate(t_mission_acc):

        x_start = 0.0

        for idx, (section_name, _, _, length, mass, CG_pos, _) in enumerate(section_missile):

            ## ----- GESTION DES MASSES ----- ##
            if section_name == "PROPERGOL ACCÉLÉRATION":
                mass_array_tot_acc[idx, t_idx] = max(0.0, mass - t_val * ma_dot)

            elif section_name in ["TUYÈRE", "RÉSERVOIR ACCÉLÉRATION"]: #"RÉSERVOIR ACCÉLÉRATION",
                mass_array_tot_acc[idx, t_idx] = mass if t_val < t_acc else 0.0

            else:
                mass_array_tot_acc[idx, t_idx] = mass

            ## ----- GESTION DU CENTRE DE GRAVITÉ ----- ##
            if section_name not in ["ENTRÉE AIR", "AILES", "QUEUE", "TUYÈRE"]:

                if section_name in ["PROPERGOL CROISIÈRE", "PROPERGOL ACCÉLÉRATION"]:

                    if section_name == "PROPERGOL CROISIÈRE":
                        deltaL_cruise = row['L_cruise_res'] - row['L_cruise_prop']
                        x_CG_prop_cruise = row['L_ogive'] + row['L_equipement'] + row['L_payload'] + deltaL_cruise + length/2
                        CG_position_acc[idx, t_idx] = mass_array_tot_acc[idx, t_idx] * x_CG_prop_cruise

                    if section_name == "PROPERGOL ACCÉLÉRATION":
                        deltaL_acc = row['L_acc_res'] - row['L_acc_prop']
                        x_CG_prop_acc = row['L_ogive'] + row['L_equipement'] + row['L_payload'] + row['L_cruise_res'] + row['L_engine_housing'] + deltaL_acc + length/2
                        CG_position_acc[idx, t_idx] = mass_array_tot_acc[idx, t_idx] * x_CG_prop_acc
                else:
                    x_CG = x_start + CG_pos

                    CG_position_acc[idx, t_idx] = mass_array_tot_acc[idx, t_idx] * x_CG

                    x_start += length

            else:
                if section_name == "ENTRÉE AIR":
                    start_AI = row['L_ogive'] + row['L_equipement'] + row['L_payload']
                    end_AI = start_AI + row['L_cruise_res'] + row['L_engine_housing'] + row['L_acc_res']
                    CD_AI = (end_AI + start_AI) / 2

                    CG_position_acc[idx, t_idx] = mass_array_tot_acc[idx, t_idx] * (start_AI + CD_AI)
                
                if section_name == "AILES":
                    start_W = row['L_ogive'] + row['L_equipement'] + row['L_payload']
                    end_W = start_AI + length
                    CD_W = (end_W + start_W) / 2

                    CG_position_acc[idx, t_idx] = mass_array_tot_acc[idx, t_idx] * (start_W + CD_W)

                if section_name == "QUEUE":
                    start_T = row['L_ogive'] + row['L_equipement'] + row['L_payload'] + row['L_cruise_res'] + row['L_engine_housing'] + row['L_acc_res'] - length
                    end_T = row['L_ogive'] + row['L_equipement'] + row['L_payload'] + row['L_cruise_res'] + row['L_engine_housing'] + row['L_acc_res']
                    CG_T = (start_T + end_T) / 2
                    
                    CG_position_acc[idx, t_idx] = mass_array_tot_acc[idx, t_idx] * (start_T + CG_T)
            
        CG_array_acc[t_idx] = CG_position_acc[:, t_idx].sum() / mass_array_tot_acc[:, t_idx].sum()

    ## ----- SIMULATION POUR LA PHASE DE CROISIÈRE ----- ##

    for t_idx, t_val in enumerate(t_mission_cruise):

        x_start = 0.0

        for idx, (section_name, _, _, length, mass, CG_pos, _) in enumerate(section_missile):
            
            ## ----- GESTION DES MASSES ----- ##
            if section_name in ["PROPERGOL ACCÉLÉRATION", "TUYÈRE", "RÉSERVOIR ACCÉLÉRATION"]: #, "RÉSERVOIR ACCÉLÉRATION"
                mass_array_tot_cruise[idx, t_idx] = 0.0
                CG_position_cruise_bladder[idx, t_idx] = 0.0
                CG_position_cruise_piston[idx, t_idx] = 0.0

            elif section_name == "PROPERGOL CROISIÈRE":
                mass_array_tot_cruise[idx, t_idx] = max(0.0, mass - 0.9 * (t_val - t_acc) * mc_dot)

            else:
                mass_array_tot_cruise[idx, t_idx] = mass

            ## ----- GESTION DU CENTRE DE GRAVITÉ ----- ##
            if section_name not in ["ENTRÉE AIR", "AILES", "QUEUE", "PROPERGOL ACCÉLÉRATION", "TUYÈRE", "RÉSERVOIR ACCÉLÉRATION"]: #, "RÉSERVOIR ACCÉLÉRATION"

                if section_name == "PROPERGOL CROISIÈRE":
                    ## ----- BLADDER ----- ##
                    deltaL_cruise = row['L_cruise_res'] - row['L_cruise_prop']
                    x_CG_prop_cruise_bladder = row['L_ogive'] + row['L_equipement'] + row['L_payload'] + deltaL_cruise + length/2
                    CG_position_cruise_bladder[idx, t_idx] = mass_array_tot_cruise[idx, t_idx] * x_CG_prop_cruise_bladder

                    ## ----- PISTON ----- ##
                    L_cp_t = length * (0.9 * (t_cruise - (t_val - t_acc))/t_cruise + 0.1)
                    deltaL_cruise_piston = row['L_cruise_res'] - L_cp_t
                    x_CG_prop_cruise_piston = row['L_ogive'] + row['L_equipement'] + row['L_payload'] + deltaL_cruise_piston + L_cp_t/2
                    CG_position_cruise_piston[idx, t_idx] = mass_array_tot_cruise[idx, t_idx] * x_CG_prop_cruise_piston
                
                else:
                    x_CG = x_start + CG_pos

                    CG_position_cruise_bladder[idx, t_idx] = mass_array_tot_cruise[idx, t_idx] * x_CG
                    CG_position_cruise_piston[idx, t_idx] = mass_array_tot_cruise[idx, t_idx] * x_CG

                    x_start += length
            else:
                if section_name == "ENTRÉE AIR":
                    start_AI = row['L_ogive'] + row['L_equipement'] + row['L_payload']
                    end_AI = start_AI + row['L_cruise_res'] + row['L_engine_housing'] + row['L_acc_res']
                    CD_AI = (end_AI + start_AI) / 2

                    CG_position_cruise_bladder[idx, t_idx] = mass_array_tot_cruise[idx, t_idx] * (start_AI + CD_AI)
                    CG_position_cruise_piston[idx, t_idx] = mass_array_tot_cruise[idx, t_idx] * (start_AI + CD_AI)
                
                if section_name == "AILES":
                    start_W = row['L_ogive'] + row['L_equipement'] + row['L_payload'] 
                    end_W = start_AI + length
                    CD_W = (end_W + start_W) / 2

                    CG_position_cruise_bladder[idx, t_idx] = mass_array_tot_cruise[idx, t_idx] * (start_W + CD_W)
                    CG_position_cruise_piston[idx, t_idx] = mass_array_tot_cruise[idx, t_idx] * (start_W + CD_W)

                if section_name == "QUEUE":
                    start_T = row['L_ogive'] + row['L_equipement'] + row['L_payload'] + row['L_cruise_res'] + row['L_engine_housing'] + row['L_acc_res'] - length
                    end_T = row['L_ogive'] + row['L_equipement'] + row['L_payload'] + row['L_cruise_res'] + row['L_engine_housing'] + row['L_acc_res']
                    CG_T = (start_T + end_T) / 2
                    
                    CG_position_cruise_bladder[idx, t_idx] = mass_array_tot_cruise[idx, t_idx] * (start_T + CG_T)
                    CG_position_cruise_piston[idx, t_idx] = mass_array_tot_cruise[idx, t_idx] * (start_T + CG_T)

        CG_array_cruise_bladder[t_idx] = CG_position_cruise_bladder[:, t_idx].sum() / mass_array_tot_cruise[:, t_idx].sum()
        CG_array_cruise_piston[t_idx] = CG_position_cruise_piston[:, t_idx].sum() / mass_array_tot_cruise[:, t_idx].sum()


    ## ----- CONCATÉNATION DES VECTEURS ----- ##
    mass_array = np.concatenate([mass_array_tot_acc, mass_array_tot_cruise[:, 1:]], axis=1)
    CG_x_bladder = np.concatenate([CG_array_acc, CG_array_cruise_bladder[1:]])
    CG_x_piston = np.concatenate([CG_array_acc, CG_array_cruise_piston[1:]])

    mass_array_2D = np.zeros(shape=t_mission_array.shape[0])
    for idx in range(t_mission_array.shape[0]):
        mass_array_2D[idx] = mass_array[:, idx].sum()

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ## ----- ÉVOLUTION DU CENTRE DE GRAVITÉ ----- ##
    axes[0, 0].set_title('Évolution du centre de gravité', fontsize=16)

    axes[0, 0].plot(t_mission_array, CG_x_bladder/diametre, c="navy", label="BLADDER TANK")
    axes[0, 0].plot(t_mission_array, CG_x_piston/diametre, c="darkorange", label="PISTON TANK")

    axes[0, 0].set_xlabel("Temps [m]")
    axes[0, 0].set_ylabel("Centre de gravité [m]")
    axes[0, 0].grid('on', alpha=0.75, linestyle="-.")
    axes[0, 0].legend(loc="upper right")

    ## ----- ÉVOLUTION DE LA MASSE ----- ##
    axes[0, 1].set_title('Évolution de la masse', fontsize=16)
    axes[0, 1].plot(t_mission_array, mass_array_2D, c='navy', label='Évolution de la masse')
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
            # x_start_T = row['L_m'] - row['L_nozzle'] - length
            x_start_T = row['L_ogive'] + row['L_payload'] + row['L_equipement'] +  row['L_cruise_res'] + row['L_engine_housing'] + row['L_acc_res'] - length
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
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()

    CGx_dict = {
        'BLADDER': CG_x_bladder,
        'PISTON': CG_x_piston
    }

    mass_dict = {
        'MASS': mass_array,
        'MASS_TOT': mass_array_2D
    }

    print(f"----- MASSE DE PROPERGOL POUR LA PHASE DE CROISIÈRE -----")
    mass_cruise_propergol = mass_array[4, :]
    # print(mass_cruise_propergol)

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

    return CGx_dict, mass_dict, t_mission_array