from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

def Get_drag_data(target_dict, diametre_dict, row, airfoil_data):

    diametre = diametre_dict['diametre']

    ## ----- Expression des vecteurs des graphiques ----- ##

        # --> sample traînée d'onde ogive
    beta_dL_sample = np.linspace(0, 2.0, 21)
    Cx_Ld_sample = np.array([1.175, 1.169, 1.15, 1.1, 1.02, 0.94, 0.885, 0.845, 0.82, 0.795, 0.78, 0.765, 0.755, 0.745, 0.74, 0.730, 0.725, 0.72, 0.71, 0.705, 0.7])

        # --> sample traînée d'onde entrée d'air
    phi_EA_sample = np.linspace(0.0, 32.0, 9)
    ratio_circ_sample = np.array([0.0, 0.0066667, 0.013333, 0.0244442, 0.035552, 0.0511106, 0.0711104, 0.0955546, 0.1133322])
    ratio_square_sample = np.array([0.0, 0.01111, 0.0244442, 0.04, 0.0577772, 0.07999992, 0.11111, 0.144443, 0.1755538])
    ratio_diamond_sample = np.array([0.0, 0.01555554, 0.033333, 0.0533328, 0.0733326, 0.099999, 0.13777764, 0.18444426, 0.2088868])

        # --> sample traînée d'onde des pièges à couche limite
    ratio_Hd_sample = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    Cx_trap_sample = np.array([np.nan, np.nan, 0.00325, 0.0038125, 0.004875, 0.006375, 0.008125, 0.010625, 0.01375, np.nan, np.nan])

        # --> sample traînée de frottement
    lam_sample = np.array([1.0, 0.9909, 0.98181, 0.97277, 0.95454, 0.93636, 0.89999, 0.88181, 0.86363, 0.82727, 0.80909, 0.78181, 0.76363])
    turb_sample = np.array([1.0, 0.981818, 0.91818, 0.82727, 0.73636, 0.64545, 0.572727, 0.49090, 0.42727, 0.3909, 0.3363, 0.299997, 0.27272])
    mach_sample = np.linspace(0, 6.0, 13)

        # --> sample coefficient de pression
    Cp_sample_press = np.array([-0.116, -0.116, -0.116, -0.22, -0.172, -0.148, -0.12, -0.1, -0.082, -0.068, -0.056, -0.048, -0.04, -0.036])
    mach_sample_press = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

        # --> abscisse vecteur
    beta_dL = np.linspace(0.0, 2.0, 500)
    phi_EA = np.linspace(0.0, 32.0, 500)
    ratio_Hd = np.linspace(0.0, 0.1, 500)
    mach_arr_frot = np.linspace(0.0, 6.0, 500)
    mach_arr_pres = np.linspace(1.0, 6.0, 450)

        # --> ordonnée vecteur
            # ogive
    interp_1 = interp1d(beta_dL_sample, Cx_Ld_sample, kind='linear')
    Cx_Ld = interp_1(beta_dL)

            # entrée d'air
    coeffs_circ = np.polyfit(phi_EA_sample, ratio_circ_sample, 5)
    coeffs_square = np.polyfit(phi_EA_sample, ratio_square_sample, 5)
    coeffs_diamond = np.polyfit(phi_EA_sample, ratio_diamond_sample, 6)
    poly_circ = np.poly1d(coeffs_circ)
    poly_square = np.poly1d(coeffs_square)
    poly_diamond = np.poly1d(coeffs_diamond)
    Cx_circ = poly_circ(phi_EA)
    Cx_square = poly_square(phi_EA)
    Cx_diamond = poly_diamond(phi_EA)

            # piège couche limite
    interp_3 = interp1d(ratio_Hd_sample, Cx_trap_sample, kind='linear')
    Cx_trap = interp_3(ratio_Hd)

            # frottement surface
    coeff_lam = np.polyfit(mach_sample, lam_sample, 3)
    fit_lam = np.poly1d(coeff_lam)
    f_lam = fit_lam(mach_arr_frot)
    coeff_turb = np.polyfit(mach_sample, turb_sample, 4)
    fit_turb = np.poly1d(coeff_turb)
    f_turb = fit_turb(mach_arr_frot)

            # pression
    arr_1, arr_2 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0], [-0.22, -0.172, -0.148, -0.12, -0.1, -0.082, -0.068, -0.056, -0.048, -0.04, -0.036]
    coeff_CP = np.polyfit(arr_1, arr_2, 4)
    Cp_fit = np.poly1d(coeff_CP)
    Mach_arr = np.concatenate([[0.0, 0.5, 1.0, 1.0], mach_arr_pres])
    Cp_arr = np.concatenate([[-0.116, -0.116, -0.116, -0.22], Cp_fit(mach_arr_pres)])

    ## ----- Target values ----- ##

        # --> ogive
    idx_beta = np.argmin(np.abs(beta_dL - target_dict['ogive']))
    Cx_wave_ogive_prod = Cx_Ld[idx_beta]

        # --> piège couche limite
    idx_hd = np.argmin(np.abs(ratio_Hd - target_dict['trap']))
    Cx_wave_trap = Cx_trap[idx_hd]

        # --> entrée d'air
    ref_ratio = np.array([1.25, 1.54, 2.0])
    ref_Cx = np.array([Cx_circ, Cx_square, Cx_diamond])
    idx_phi = np.argmin(np.abs(phi_EA - target_dict['phi']))
    idx_insert = np.searchsorted(ref_ratio, target_dict['ratio'])
    Cx_interp = ((target_dict['ratio'] - ref_ratio[idx_insert - 1]) / (ref_ratio[idx_insert] - ref_ratio[idx_insert - 1])) * (ref_Cx[idx_insert][idx_phi] - ref_Cx[idx_insert - 1][idx_phi]) + ref_Cx[idx_insert - 1][idx_phi]

        # --> frottement
    idx_mach_reynolds = np.argmin(np.abs(mach_arr_frot - target_dict['mach']))
    f_lam_val = f_lam[idx_mach_reynolds]
    f_turb_val = f_turb[idx_mach_reynolds]

        # --> pression
    idx_mach = np.argmin(np.abs(mach_arr_pres - target_dict['mach']))
    Cp_culot = Cp_arr[idx_mach]

    graph_label = {
        'ax0': {
            'x_label': r'$\beta \frac{d}{L_{o}}$',
            'y_label': r'$C_x \left(\frac{L_o}{d}\right)^2$',
            'title': r'Traînée d onde ogive'
        },

        'ax1': {
            'x_label': r'$\frac{H}{d}$',
            'y_label': r'$C_{x, trap}$',
            'title': r'Traînée des 4 pièges'
        },

        'ax2': {
            'x_label': r'$Mach$',
            'y_label': r'$f$',
            'title': r'Coefficient de frottement'
        },

        'ax3': {
            'x_label': r'$Mach$',
            'y_label': r'$K_{p_c}$',
            'title': r'Coefficient de pression de culot'
        },

        'mergedax': {
            'x_label': r'$\psi_e$',
            'y_label': r'$C_x (A_m)$',
            'title': r'Traînée d onde entrée d air'
        },
    }

    sample_data = {

        'ax0': {
            'scat1': [beta_dL_sample, Cx_Ld_sample, 'o', 'red', 'Données ogive']
        },

        'ax1': {
            'scat1': [ratio_Hd_sample, Cx_trap_sample, 'o', 'red', 'Données piège']
        },

        'ax2': {
            'scat1': [mach_sample, lam_sample, 'o', 'red', 'Données laminaire'],
            'scat2': [mach_sample, turb_sample, 's', 'magenta', 'Données turbulent']
        },

        'ax3': {
            'scat1': [arr_1, arr_2, 'o', 'red', 'Données coeff pression']
        },

        'mergedax': {
            'scat1': [phi_EA_sample, ratio_circ_sample,'o', 'red', r'$\frac{A_m}{A_1} = 1.25$'],
            'scat2': [phi_EA_sample, ratio_square_sample,'s', 'magenta', r'$\frac{A_m}{A_1} = 1.54$'],
            'scat3': [phi_EA_sample, ratio_diamond_sample, 'D', 'darkorange', r'$\frac{A_m}{A_1} = 2$']
        },
    }

    plot_data = {
        'ax0': {
            'plot1': [beta_dL, Cx_Ld]
        },

        'ax1': {
            'plot1': [ratio_Hd, Cx_trap]
        },

        'ax2': {
            'plot1': [mach_arr_frot, f_lam],
            'plot2': [mach_arr_frot, f_turb]
        },

        'ax3': {
            'plot1': [Mach_arr, Cp_arr]
        },

        'mergedax': {
            'plot1': [phi_EA, Cx_circ],
            'plot2': [phi_EA, Cx_square],
            'plot3': [phi_EA, Cx_diamond]
        }
    }

    valid_data = {
        'ax0': {
            'sol1': [target_dict['ogive'], Cx_wave_ogive_prod, '#AF1740', r"$C_x \left(\frac{L_o}{d}\right)^2$ = " + str(Cx_wave_ogive_prod), Cx_wave_ogive_prod, beta_dL.min(), beta_dL[idx_beta] + 0.05, target_dict['ogive'], Cx_Ld.min(), Cx_Ld[idx_beta] + 0.02]
        },

        'ax1': {
            'sol1': [target_dict['trap'], Cx_wave_trap, '#AF1740', r"$C_{x, trap}$ = " + str(Cx_wave_trap), Cx_wave_trap, ratio_Hd.min(), ratio_Hd[idx_hd] + 0.01, target_dict['trap'], np.nan_to_num(np.nanmin(Cx_trap), nan=0.0), Cx_trap[idx_hd] + 0.001]
        },

        'ax2': {
            'sol1': [target_dict['mach'], f_lam_val, '#AF1740', r"$f_{lam}$ = " + str(f_lam_val), f_lam_val, mach_arr_frot.min(), mach_arr_frot[idx_mach_reynolds] + 0.01, target_dict['mach'], f_lam.min(), f_lam[idx_mach] + 0.05],
            'sol2': [target_dict['mach'], f_turb_val, '#5D8736',  r"$f_{turb}$ = " + str(f_turb_val), f_turb_val, mach_arr_frot.min(), mach_arr_frot[idx_mach_reynolds] + 0.01, target_dict['mach'], f_turb.min(), f_turb[idx_mach] + 0.01]
        },

        'ax3': {
            'sol1': [target_dict['mach'], Cp_culot, '#AF1740', r"$C_{p}$ = " + str(Cp_culot), Cp_culot, Mach_arr.min(), Mach_arr[idx_mach] + 0.01, target_dict['mach'], Cp_arr.max(), Cp_arr[idx_mach] - 0.025]
        },

        'mergedax': {
            'sol1': [target_dict['phi'], Cx_interp, '#AF1740', r"$C_{x}^{(A_m)}$ = " + str(Cx_interp), Cx_interp, phi_EA.min(), phi_EA[idx_phi] + 2, phi_EA[idx_phi], 0, Cx_interp + 0.0125]
        }
    }

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle('Données relatives à l expression de la traînée du missile', fontsize=20)
    gs = GridSpec(2, 3, figure=fig)

    axis = []
    axis.append(fig.add_subplot(gs[0, 0])) 
    axis.append(fig.add_subplot(gs[0, 1])) 
    axis.append(fig.add_subplot(gs[1, 0]))
    axis.append(fig.add_subplot(gs[1, 1])) 
    axis.append(fig.add_subplot(gs[:, 2]) )

    for (idx, ax), label_graph, sample, plot, valid in zip(enumerate(axis), graph_label.values(), sample_data.values(), plot_data.values(), valid_data.values()):

        if idx == 3:
            ax.invert_yaxis()
            
        ## ----- configuration du scatter ----- ##,
        for scat in sample.values():
            ax.scatter(scat[0], scat[1], marker=scat[2], edgecolor=scat[3], facecolor='none', label=scat[4])

        ## ----- configuration bonne valeur ----- ##
        for sol in valid.values():
            ax.scatter(sol[0], sol[1], c=sol[2], marker='*', s=200, label=sol[3])
            ax.hlines(y=sol[4] ,xmin=sol[5] , xmax=sol[6] , color='#8B5DFF', linestyle='--', alpha=0.75)
            ax.vlines(x=sol[7] ,ymin=sol[8] , ymax=sol[9] , color='#8B5DFF', linestyle='--', alpha=0.75)

        ## ----- configuration du plot ----- ##
        for graph in plot.values():
            ax.plot(graph[0], graph[1], linestyle='-.', color='gray')

        ## ----- configuration du graphique ----- ##
        ax.set_title(label_graph['title'])
        ax.set_xlabel(label_graph['x_label'])
        ax.set_ylabel(label_graph['y_label'])
        ax.legend(loc='best', fontsize=10)
        ax.grid('on', alpha=0.75)

    plt.show()

    ## ----- Return Cx wave value -- ##

    internal_diameter = diametre_dict['internal_diameter']
    thickness = diametre_dict['thickness']
    beta = np.sqrt(target_dict['mach']**2 - 1)

    A_m = 0.25 * np.pi * (internal_diameter + 2*thickness)**2
    A_ref = 0.25 * np.pi * diametre**2

    Cx_wave_ogive = Cx_wave_ogive_prod * (diametre/row['L_ogive'])**2
    Cx_wave_EA = 4 * Cx_interp * (A_m/A_ref)
    Cx_wave_trap = Cx_trap[idx_hd]
    Cx_wing = 4 * (airfoil_data['aile']['K']/beta) * (airfoil_data['aile']['ratio_ec']**2) * airfoil_data['aile']['S']
    Cx_tail = 4 * (airfoil_data['gouverne']['K']/beta) * (airfoil_data['aile']['ratio_ec']**2) * airfoil_data['gouverne']['S']

    drag_wave = {
        'Cx_ogive': Cx_wave_ogive,
        'Cx_trap': Cx_wave_trap,
        'Cx_EA': Cx_wave_EA,
        'Cx_wing': Cx_wing,
        'Cx_tail': Cx_tail
    }

    Cx_wave_tot = np.sum([val for val in drag_wave.values()])
    
    drag_dict = {
        'Cx_wave': Cx_wave_tot
    }

    return drag_wave, drag_dict, Cp_culot