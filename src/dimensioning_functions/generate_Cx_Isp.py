import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

def generate_Cx_Isp(Mach_cruise, shape, img_path):

    # Données pour le Cx en fonction du Mach
    Mach_acc_dis = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    Cx_acc_dis = np.array([0.4, 0.4, 0.4, 0.415, 0.455, 0.75, 0.79, 0.74, 0.69, 0.645, 0.585])
    interpolation_acc = interp1d(Mach_acc_dis, Cx_acc_dis, kind='linear')
    Mach_acc_arr = np.linspace(Mach_acc_dis.min(), Mach_acc_dis.max(), shape)
    Cx_acc_arr = interpolation_acc(Mach_acc_arr)

    Mach_cruise_dis = np.array([1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    Cx_cruise_dis = np.array([0.515, 0.475, 0.447, 0.42, 0.395, 0.38, 0.365])
    interpolation_cruise = interp1d(Mach_cruise_dis, Cx_cruise_dis, kind='linear')
    Mach_cruise_arr = np.linspace(Mach_cruise_dis.min(), Mach_cruise_dis.max(), shape)
    Cx_cruise_arr = interpolation_cruise(Mach_cruise_arr)

    idx = np.argmin(np.abs(Mach_cruise_arr - 2.0))
    Cd_c = Cx_cruise_arr[idx]

    # Données pour l'Isp en fonction du Mach
    Mach_kerosene_dis = np.array([1.9, 3.1])
    Isp_kerosene_dis = np.array([1475, 1175])
    Mach_liquide_dense_dis = np.array([1.85, 3.175])
    Isp_liquide_dense_dis = np.array([1440, 1120])
    Mach_bore_dis = np.array([1.825, 3.125])
    Isp_bore_dis = np.array([1075, 825])
    Mach_hydrocarbure_dis = np.array([1.825, 3.175])
    Isp_hydrocarbure_dis = np.array([975, 750])

    # Interpolation
    nbr_point = 1000
    Mach_kerosene = np.linspace(Mach_kerosene_dis[0], Mach_kerosene_dis[1], nbr_point)
    Isp_kerosene = np.linspace(Isp_kerosene_dis[0], Isp_kerosene_dis[1], nbr_point)
    Mach_liquide = np.linspace(Mach_liquide_dense_dis[0], Mach_liquide_dense_dis[1], nbr_point)
    Isp_liquide = np.linspace(Isp_liquide_dense_dis[0], Isp_liquide_dense_dis[1], nbr_point)
    Mach_bore = np.linspace(Mach_bore_dis[0], Mach_bore_dis[1], nbr_point)
    Isp_bore = np.linspace(Isp_bore_dis[0], Isp_bore_dis[1], nbr_point)
    Mach_hydrocarbure = np.linspace(Mach_hydrocarbure_dis[0], Mach_hydrocarbure_dis[1], nbr_point)
    Isp_hydrocarbure = np.linspace(Isp_hydrocarbure_dis[0], Isp_hydrocarbure_dis[1], nbr_point)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Paramètres de la simulation", fontsize=16)

    axes[0].scatter(Mach_acc_dis, Cx_acc_dis, s=40, marker="*", color="navy", label="discrete acc data")
    axes[0].scatter(Mach_cruise_dis, Cx_cruise_dis, s=40, marker="*", color="red", label="discrete cruise data")
    axes[0].plot(Mach_acc_arr, Cx_acc_arr, color="blue", linestyle="--", label="Cx acc")
    axes[0].plot(Mach_cruise_arr, Cx_cruise_arr, color="darkorange", linestyle="--", label="Cx cruise")
    axes[0].grid(True, alpha=0.5, linestyle='--')
    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("Mach")
    axes[0].set_ylabel(r"$C_x$")
    axes[0].set_ylim([0.0, 1.0])
    axes[0].set_xticks(np.arange(0, 3.2, 0.2))
    axes[0].set_yticks(np.arange(0, 1.1, 0.1))
    axes[0].set_title("Loi des $C_x$ en fonction du Mach")

    MACH_arr = [Mach_kerosene, Mach_liquide, Mach_bore, Mach_hydrocarbure]
    ISP_arr = [Isp_kerosene, Isp_liquide, Isp_bore, Isp_hydrocarbure]
    LABEL_arr = ["kérosène, d = 0.8", "liquide dense, d = 1.0", "dopé au bore, d = 1.6", "hydrocarbure, d = 1.3"]

    for mach, isp, label in zip(MACH_arr, ISP_arr, LABEL_arr):
        axes[1].plot(mach, isp, label=label)
    axes[1].vlines(x=2.0, ymin=600, ymax=1600, linestyle="--", color="#AF1740", label="Mach de croisière")
    axes[1].grid(True, alpha=0.5, linestyle="--")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim([1.6, 3.4])
    axes[1].set_ylim([600, 1600])
    axes[1].set_xlabel("Mach")
    axes[1].set_ylabel("Isp")
    axes[1].set_title("Isp en fonction du Mach et du comburant")

    fig_path = os.path.join(img_path, "Parametres de simulation.jpg")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', quality=95)
    plt.show()

    isp_value = np.zeros(4)
    densite_value = [800, 1000, 1600, 1300]
    for idx, (mach_arr, isp_array) in enumerate(zip(MACH_arr, ISP_arr)):
        isp_idx = np.argmin(np.abs(mach_arr - Mach_cruise))
        isp_value[idx] = np.round(isp_array[isp_idx], 2)

    # stockage des valeurs
    keys_dict_sub = np.array(["Impulsion Spécifique", "Masse volumique"])
    values_dict_sub = np.array([(isp, rho) for isp, rho in zip(isp_value, densite_value)])

    keys_dict_main = np.array(["Kerosene", "Liquide dense", "Bore", "Hydrocarbure"])

    cruise_data_dict = {
        key: {
            keys_dict_sub[0]: values[0], 
            keys_dict_sub[1]: values[1], 
        }
        for key, values in zip(keys_dict_main, values_dict_sub)
    }

    return keys_dict_main, cruise_data_dict, Cx_acc_arr, Mach_acc_arr, isp_value, Cd_c, densite_value