import numpy as np
from scipy.integrate import trapz


def constraints_algo(time, data_mission, i, c, graph_component, iterate):

    ## ----- GESTION DES ARGUMENTS ----- ##
    t_acc = time['t_acc']
    t_cruise = time['t_cruise']

    i_a = i['i_a']
    i_c = i['i_c']

    c_a = c['c_a']
    c_c = c['c_c']

    gamma = data_mission['gamma']
    P_amb = data_mission['P_amb']
    Mach_cruise = data_mission['Mach_cruise']
    g0 = data_mission['g0']
    Cd_c = data_mission['Cd_c']
    Delta_Vr = data_mission['Delta_Vr']
    Isp_acc = data_mission['Isp_acc']

    m_ogive = data_mission['mass']['m_ogive']
    m_equipement = data_mission['mass']['m_equipement']
    m_tails = data_mission['mass']['m_tails']
    m_payload = data_mission['mass']['m_payload']
    m_engine_h = data_mission['mass']['m_engine_h']
    m_nozzle = data_mission['mass']['m_nozzle']

    rho_payload = data_mission['rho']['rho_payload']
    rho_equipement = data_mission['rho']['rho_equipement']
    rho_c = data_mission['rho']['rho_c']
    rho_a = data_mission['rho']['rho_a']

    Mach_acc_arr = graph_component['Mach_acc_arr']
    Cx_acc_arr = graph_component['Cx_acc_arr']

    keys_dict_main = iterate['keys_dict_main']
    cruise_data_dict = iterate['cruise_data_dict']

    ## ----- VECTEUR ----- ##
    dt = 0.001
    d_missile = np.arange(0.2, 0.6 + dt, dt)
    shape = d_missile.shape[0]

    t_acc_arr = np.linspace(0, t_acc, shape)

    ## ----- FUNCTION ----- ##
    m_cp_fnc = lambda d_m, Isp_c: 1.1 * np.divide(np.pi * gamma * P_amb * Cd_c * Mach_cruise**2 * d_m**2, 8 * Isp_c * g0) * t_cruise

    m_r_fnc = lambda i, m_p: np.divide(i * m_p, 1 - i)

    alpha_fnc = lambda K_val: np.divide((1 + K_val) * Delta_Vr, Isp_acc * g0)

    m_t_fnc = lambda alpha_val, sum_mass: sum_mass / (1 - np.divide(np.exp(alpha_val) - 1, np.exp(alpha_val)) * np.divide(1, 1 - i_a)) 

    m_ap_fnc = lambda alpha_val, m_t_val: np.divide(np.exp(alpha_val) - 1, np.exp(alpha_val)) * m_t_val

    L_prop_fnc = lambda m_i, rho_i, d_m: np.divide(4 * m_i, np.pi * rho_i * d_m**2)

    L_res_fnc = lambda m_i, rho_i, d_m, c_i: np.divide(4 * m_i, np.pi * rho_i * d_m**2 * c_i)

    def K_prime_expression(d_m, mass_fnc_t):
        cst_term = (np.pi * gamma * P_amb * d_m**2) / (8 * Delta_Vr)
        integrand = (Cx_acc_arr * Mach_acc_arr**2) / mass_fnc_t
        K_prime = cst_term * trapz(integrand, t_acc_arr)
        return K_prime

    ## ----- CODE ----- ##
    nbr_oxydizer = keys_dict_main.shape[0]

    mass_tensor = np.zeros((nbr_oxydizer, shape, 7))
    length_tensor = np.zeros((nbr_oxydizer, shape, 11))
    K_matrix = np.zeros((nbr_oxydizer, shape))

    for idx_deph, data in enumerate(cruise_data_dict.values()):

        Isp_cruise = data["Impulsion Spécifique"]
        rho_cruise = data["Masse volumique"]

        for idx_row, d_m in enumerate(d_missile):

            ## ----- CONSTNATES DE LA SIMULATION ----- ##
            K = 0.1
            K_vec = [K]
            eps = 1
            count = 0

            ## ----- CALCUL DES MASSES ----- ##
                # masses PDF:ELISA_3
            # m_intakes = 35 + 250 * d_m**2
            # m_wings = 25 + 250 * d_m**2
                # masses PDF:projet_missile_stato
            m_intakes = 35 + 250 * d_m**2
            m_wings = 25 + 285 * d_m**2

            m_tot_empty = np.sum([m_ogive, m_equipement, m_tails, m_payload, m_engine_h, m_nozzle, m_intakes, m_wings])
            m_cp = m_cp_fnc(d_m=d_m, Isp_c=Isp_cruise)
            m_cr = m_r_fnc(i=i_c, m_p=m_cp)
            m_tot_cruise = m_tot_empty + m_cp + m_cr

            ## ----- CONVERGENCE AVEC PHASE ACCELERATION ----- ##
            while np.abs(eps) > 1e-5:
                
                K_prime = K_vec[-1]

                alpha = alpha_fnc(K_val=K_vec[-1])

                m_t = m_t_fnc(alpha_val=alpha, sum_mass=m_tot_cruise)

                m_ap = m_ap_fnc(alpha_val=alpha, m_t_val=m_t)

                m_ar = m_r_fnc(i=i_a, m_p=m_ap)

                m_tot = m_tot_cruise + m_ap + m_ar

                M_fnc_t = m_t - m_ap * (t_acc_arr/t_acc)

                K_prime = K_prime_expression(d_m=d_m, mass_fnc_t=M_fnc_t)
                K_vec.append(K_prime)
                
                eps = K_vec[-1] - K_vec[-2]
            
            ## ----- CALCUL DES LONGUEURS CARACTÉRISTIQUES DU MISSILE ----- ##
            L_acc_noz = 0.5 * d_m
            L_ogive = 3 * d_m
            L_payload = L_prop_fnc(m_i=m_payload, rho_i=rho_payload, d_m=d_m)
            L_equipement = L_prop_fnc(m_i=m_equipement, rho_i=rho_equipement, d_m=d_m)
            L_engine_housing = 0.5 * d_m
            L_cr = L_res_fnc(m_i=m_cp, rho_i=rho_c, d_m=d_m, c_i=c_c)
            L_ar = L_res_fnc(m_i=m_ap, rho_i=rho_a, d_m=d_m, c_i=c_a)

            L_AirIntakes = L_cr + L_engine_housing + L_ar
            length_tensor[idx_deph, idx_row, 8] = L_AirIntakes

            L_tails = d_m
            length_tensor[idx_deph, idx_row, 9] = L_tails

            L_Wings = L_cr
            length_tensor[idx_deph, idx_row, 10] = L_Wings

            ## ----- ASSIGNATION DES VALEURS DES LONGUEURS ----- ##
            length_tensor[idx_deph, idx_row, :7] = np.array([L_ogive, L_equipement, L_payload, L_engine_housing, L_cr, L_ar, L_acc_noz])
            length_tensor[idx_deph, idx_row, 7] = np.sum(length_tensor[idx_deph, idx_row, :7])

            ## ----- ASSIGNATION DES VALEURS DES MASSES ----- ##
            mass_tensor[idx_deph, idx_row, :6] = np.array([m_cp, m_cr, m_ap, m_ar, m_intakes, m_wings])
            mass_tensor[idx_deph, idx_row, 6] = mass_tensor[idx_deph, idx_row, :4].sum() + m_tot_empty

            ## ----- ASSIGNATION DE LA VALEUR DE K ----- ##
            K_matrix[idx_deph, idx_row] = K_vec[-1] 

    return mass_tensor, length_tensor, d_missile