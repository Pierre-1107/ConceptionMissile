import numpy as np
import pandas as pd
import os

def generate_csv_txt(cruise_data_dict, d_missile, data_mission, length_tensor, mass_tensor, isp_value, densite_value, resultats_path, c):

    shape = d_missile.shape[0]

    c_a = c['c_a']
    c_c = c['c_c']


    m_ogive = data_mission['mass']['m_ogive']
    m_equipement = data_mission['mass']['m_equipement']
    m_tails = data_mission['mass']['m_tails']
    m_payload = data_mission['mass']['m_payload']
    m_engine_h = data_mission['mass']['m_engine_h']
    m_nozzle = data_mission['mass']['m_nozzle']

    for keys_idx, key in enumerate(cruise_data_dict.keys()):
        
        data = {
            ## ----- caractéristiques du missiles ----- ##
            "d_m": d_missile,
            "L_m": length_tensor[keys_idx, :, 7],
            "m_m": mass_tensor[keys_idx, :, 6],

            ## ----- masses ----- ##
            "m_ogive": np.full(shape, m_ogive),
            "m_equipement": np.full(shape, m_equipement),
            "m_payload": np.full(shape, m_payload),
            "m_engine_housing": np.full(shape, m_engine_h),
            "m_nozzle": np.full(shape, m_nozzle),
            "m_Tails": np.full(shape, m_tails),
            "m_cruise_prop": mass_tensor[keys_idx, :, 0],
            "m_cruise_res": mass_tensor[keys_idx, :, 1],
            "m_acc_prop": mass_tensor[keys_idx, :, 2],
            "m_acc_res": mass_tensor[keys_idx, :, 3],
            "m_AirIntakes": mass_tensor[keys_idx, :, 4],
            "m_Wings": mass_tensor[keys_idx, :, 5],

            ## ----- longueurs ----- ##
            "L_ogive": length_tensor[keys_idx, :, 0],
            "L_equipement": length_tensor[keys_idx, :, 1],
            "L_payload": length_tensor[keys_idx, :, 2],
            "L_engine_housing": length_tensor[keys_idx, :, 3],
            "L_nozzle": length_tensor[keys_idx, :, 6],
            "L_cruise_res": length_tensor[keys_idx, :, 4],
            "L_acc_res": length_tensor[keys_idx, :, 5],
            "L_cruise_prop": c_c * length_tensor[keys_idx, :, 4],
            "L_acc_prop": c_a * length_tensor[keys_idx, :, 5],
            "L_AirIntakes": length_tensor[keys_idx, :, 8],
            "L_Tails": length_tensor[keys_idx, :, 9],
            "L_Wings": length_tensor[keys_idx, :, 10],

            ## ----- Caractéristiques Propulsives ----- ##
            "Impulsion spécifique": np.full(shape, isp_value[keys_idx]),
            "Masse volumique": np.full(shape, densite_value[keys_idx])
        }

        data_df = pd.DataFrame(data)
        df_path = os.path.join(resultats_path, f"{key}.csv")
        data_df.to_csv(df_path, index=False, sep="\t", float_format="%.5f")

    return