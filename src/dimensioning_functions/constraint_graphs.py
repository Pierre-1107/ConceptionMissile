import matplotlib.pyplot as plt
import numpy as np
import os

def constraint_graphs(choosen_oxydiser, keys_dict_main, length_tensor, mass_tensor, d_missile, isp_value, cruise_data_dict, img_path):

    idx_choosen_oxydiser = np.where(keys_dict_main == choosen_oxydiser)[0][0]
    max_length = 6
    max_ratio = 16.5

    fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    fig.suptitle("Caractéristiques du missile", fontsize=24)

    # Définir les couleurs et styles
    main_color = 'navy'
    highlight_color = '#8B5DFF'
    acceptable_color = '#A1EEBD'
    limit_color = '#AF1740'

    # Pré-calculs
    ratio = length_tensor[idx_choosen_oxydiser, :, 7] / d_missile
    chosen_length = length_tensor[idx_choosen_oxydiser, :, 7]
    chosen_mass = mass_tensor[idx_choosen_oxydiser, :, 6]

    acceptable_Lm = np.where(length_tensor[idx_choosen_oxydiser, :, 7] < max_length)
    acceptable_ratio = np.where(ratio < max_ratio)

    idx_max_length = np.argmin(np.abs(length_tensor[idx_choosen_oxydiser, :, 7] - max_length))
    idx_max_ratio = np.argmin(np.abs(ratio - max_ratio))

    limite_idx = np.max(a=np.array([idx_max_length, idx_max_ratio]))
    acceptable_mass = np.where(chosen_mass > chosen_mass[limite_idx])

    # Liste des titres et étiquettes
    titles = [
        "Différents oxydants",
        f'Oxydant : {choosen_oxydiser}, Isp = {isp_value[idx_choosen_oxydiser]} s',
        "",
        "",
        "",
        "",
    ]

    ylabels = [
        r"Longueur du missile : $L_m$ (m)",
        "",
        r"Ratio : $L_m/D_m$",
        "",
        r"Masse du missile : $m_m$ (kg)",
        "" 
    ]

    for idx, ax in enumerate(axs.flat):

        if idx == 0:
            for key_idx, key in enumerate(cruise_data_dict.keys()):
                ax.plot(d_missile, length_tensor[key_idx, :, 7], label=f'{key}, Isp = {isp_value[key_idx]} s')
            ax.hlines(y=max_length, xmin=d_missile.min(), xmax=d_missile.max(), color=limit_color, linestyle="-.", label=fr"Longueur limite, $L_m$ = {max_length}")

        elif idx == 1:
            ax.plot(d_missile, chosen_length, color=main_color, linestyle='--', label=r"$L_m$")
            ax.hlines(y=max_length, xmin=d_missile.min(), xmax=d_missile.max(), color=limit_color, linestyle='-.', label=fr"Longueur limite, $L_m$ = {max_length} m")
            ax.vlines(x=d_missile[idx_max_length], ymin=chosen_length.min(), ymax=chosen_length.max(), color=highlight_color, linestyle='-.', label=fr"Diamètre limite, $d_m$ = {d_missile[idx_max_length]:.3f} m")
            ax.scatter(d_missile[acceptable_Lm], chosen_length[acceptable_Lm], color=acceptable_color, s=10, marker='o', label="Valeurs acceptables")

        elif idx == 2:
            for key_idx, key in enumerate(cruise_data_dict.keys()):
                ax.plot(d_missile, length_tensor[key_idx, :, 7] / d_missile, label=f'{key}, Isp = {isp_value[key_idx]} s')
            ax.hlines(y=max_ratio, xmin=d_missile.min(), xmax=d_missile.max(), color=limit_color, linestyle="-.", label=fr"Ratio limite, $L_m/d_m$ = {max_ratio}")

        elif idx == 3:
            ax.plot(d_missile, ratio, color=main_color, linestyle='--', label=r"$\frac{L_m}{d_m}$")
            ax.hlines(y=max_ratio, xmin=d_missile.min(), xmax=d_missile.max(), color=limit_color, linestyle='-.', label=fr"Ratio limite, $L_m/d_m$ = {max_ratio}")
            ax.vlines(x=d_missile[idx_max_ratio], ymin=ratio.min(), ymax=ratio.max(), color=highlight_color, linestyle='-.', label=fr"Diamètre limite, $d_m$ = {d_missile[idx_max_ratio]:.3f} m")
            ax.scatter(d_missile[acceptable_ratio], ratio[acceptable_ratio], color=acceptable_color, s=10, marker='o', label="Valeurs acceptables")

        elif idx == 4:
            for key_idx, key in enumerate(cruise_data_dict.keys()):
                ax.plot(d_missile, mass_tensor[key_idx, :, 6], label=f'{key}, Isp = {isp_value[key_idx]} s')
            ax.set_xlabel(r"$d_m$ (m)")

        elif idx == 5:
            ax.plot(d_missile, chosen_mass, color=main_color, linestyle='--', label=r"$m_m$")
            ax.vlines(x=d_missile[limite_idx], ymin=chosen_mass.min(), ymax=chosen_mass.max(), color=highlight_color, linestyle='-.', label=fr"Diamètre limite, $d_m$ = {d_missile[limite_idx]:.3f} m")
            ax.hlines(y=chosen_mass[limite_idx], xmin=d_missile.min(), xmax=d_missile.max(), color=limit_color, linestyle='-.', label=fr"Masse limite, $m_m$ = {chosen_mass[limite_idx]:.3f} kg")
            ax.scatter(d_missile[acceptable_mass], chosen_mass[acceptable_mass], color=acceptable_color, s=10, marker='o', label="Valeurs acceptables")
            ax.set_xlabel(r"$d_m$ (m)")

        ax.set_title(titles[idx])
        ax.set_ylabel(ylabels[idx])
        ax.legend(loc='upper right')
        ax.grid("on", alpha=0.5, linestyle="-.")

    fig_path = os.path.join(img_path, "Caracteristiques_(3-2).jpg")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', quality=95)
    plt.show()
