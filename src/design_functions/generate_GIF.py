import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import numpy as np


def GenerateGIF(CGx_dict, mass_dict, t_tot_array, diametre, row, section_missile, img_path):
    """
    Génère une animation GIF représentant l'évolution du centre de gravité (CG), de la masse, et la géométrie d'un missile.

    Cette fonction crée une animation en deux parties :
    1. Graphiques de l'évolution du centre de gravité et de la masse en fonction du temps.
    2. Vue géométrique détaillée du missile avec ses différentes sections.

    Args:
        CGx_dict (dict): Dictionnaire contenant les coordonnées du centre de gravité (CG) pour le piston et le bladder.
                        Clés attendues : "PISTON", "BLADDER".
        mass_dict (dict): Dictionnaire contenant les données de masse du missile.
                        Clé attendue : "MASS_TOT".
        t_tot_array (numpy.ndarray): Tableau des instants de temps total simulés.
        diametre (float): Diamètre du missile.
        row (dict): Dictionnaire contenant des dimensions spécifiques du missile (longueur totale, ogive, coiffe, etc.).
        section_missile (list): Liste décrivant les sections du missile. Chaque élément est un tuple contenant :
                                (nom de la section, coordonnées x, coordonnées y, longueur, autres dimensions, couleur).
        img_path (str): Chemin d'accès où sauvegarder l'animation GIF générée.

    Returns:
        None: La fonction sauvegarde le fichier GIF dans le chemin spécifié et affiche les graphiques.
    """
    PISTON_CG = CGx_dict["PISTON"].copy()
    BLADDER_CG = CGx_dict["BLADDER"].copy()
    MASS = mass_dict["MASS_TOT"].copy()
    T_TOT = t_tot_array.copy()

    step = 20
    BLADDER_CG_sampled = BLADDER_CG[::step]
    PISTON_CG_sampled = PISTON_CG[::step]
    MASS_SAMPLED = MASS[::step]
    t_array = T_TOT[::step]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    fig.delaxes(ax=axes[1, 0])
    fig.delaxes(ax=axes[1, 1])

    line_bladder, = axes[0, 0].plot([], [], c='navy', label=f"Bladder CG = {BLADDER_CG_sampled[0]:.3f} m")
    line_piston, = axes[0, 0].plot([], [], c='darkorange', label=f"Piston CG = {PISTON_CG_sampled[0]:.3f} m")

    axes[0, 0].set_title('Évolution du centre de gravité', fontsize=16)
    axes[0, 0].grid('on', linestyle='-.', alpha=0.75)
    axes[0, 0].set_xlabel('Temps [s]')
    axes[0, 0].set_ylabel('Centre de gravité [m]')
    axes[0, 0].set_xlim(0, t_array[-1])
    axes[0, 0].set_ylim(
        min(BLADDER_CG_sampled.min(), PISTON_CG_sampled.min()) - 0.1,
        max(BLADDER_CG_sampled.max(), PISTON_CG_sampled.max()) + 0.1,
    )
    axes[0, 0].legend()


    line_mass, = axes[0, 1].plot([], [], c='navy', label=f"Masse = {MASS_SAMPLED[0]:.3f} kg")

    axes[0, 1].set_title('Évolution de la masse', fontsize=16)
    axes[0, 1].grid('on', linestyle='-.', alpha=0.75)
    axes[0, 1].set_xlabel('Temps [s]')
    axes[0, 1].set_ylabel('Masse [kg]')
    axes[0, 1].set_xlim(0, t_array[-1])
    axes[0, 1].set_ylim(MASS_SAMPLED.min() - 0.1, MASS_SAMPLED.max() + 0.1)
    axes[0, 1].legend()

    def init():
        line_bladder.set_data([], [])
        line_piston.set_data([], [])
        line_mass.set_data([], [])
        return line_bladder, line_piston, line_mass

    def update_data(frame):
        fig.suptitle(f"Animation au temps : {t_array[frame]:.3f} s", fontsize=18)
        
        line_bladder.set_data(t_array[:frame], BLADDER_CG_sampled[:frame])
        line_piston.set_data(t_array[:frame], PISTON_CG_sampled[:frame])
        line_bladder.set_label(f"Bladder CG = {BLADDER_CG_sampled[frame]:.3f} m")
        line_piston.set_label(f"Piston CG = {PISTON_CG_sampled[frame]:.3f} m")
        axes[0, 0].legend()

        line_mass.set_data(t_array[:frame], MASS_SAMPLED[:frame])
        line_mass.set_label(f"Masse = {MASS_SAMPLED[frame]:.3f} kg") 
        axes[0, 1].legend()

        return line_bladder, line_piston, line_mass

    animation = FuncAnimation(
        fig=fig,
        func=update_data,
        init_func=init,
        frames=len(t_array),
        interval=20,
        repeat=False,
    )

    merged_axes = fig.add_subplot(2, 1, 2)  
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

    anim_path = os.path.join(img_path, "animation.gif")
    animation.save(anim_path, writer=PillowWriter(fps=60))
    plt.show()