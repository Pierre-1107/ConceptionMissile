import numpy as np
from typing import List

def generate_missile_section(row, x_ogive, y_ogive, x_CG_ogive) -> List:

    """
    Génère une liste des sections d'un missile avec leurs caractéristiques.

    Arguments :
        row : dict
            Dictionnaire contenant les dimensions et masses des différentes sections du missile.
        x_ogive : array-like
            Vecteur des coordonnées x pour l'ogive.
        y_ogive : fonction
            Fonction pour calculer les coordonnées y en fonction de x.
        x_CG_ogive : float
            Position du centre de gravité de l'ogive.

    Retourne :
        list
            Une liste contenant les caractéristiques de chaque section du missile.

    Fonctionnement des tuples :
        Chaque élément de la liste retournée est un tuple décrivant une section spécifique du missile.
        Exemple pour l'ogive :
            ("OGIVE", x_ogive, y_ogive(x_ogive), row['L_ogive'], row['m_ogive'], x_CG_ogive, '#686D76')

            - "OGIVE" : Nom de la section.
            - x_ogive : Coordonnées x représentant la géométrie de l'ogive.
            - y_ogive(x_ogive) : Coordonnées y calculées dynamiquement à partir des coordonnées x via une fonction.
            - row['L_ogive'] : Longueur de l'ogive, extraite du dictionnaire `row`.
            - row['m_ogive'] : Masse de l'ogive, extraite du dictionnaire `row`.
            - x_CG_ogive : Position du centre de gravité de l'ogive.
            - '#686D76' : Code couleur associé à la section pour des fins de visualisation.
    """

    size = 100

    section_missile = [

    ("OGIVE", x_ogive, y_ogive(x_ogive), row['L_ogive'], row['m_ogive'], x_CG_ogive, '#686D76'),
    ("CHARGE", np.full(size, np.nan), np.full(size, np.nan), row['L_payload'], row['m_payload'], row['L_payload']/2, '#A7E6FF'),
    ("EQUIPEMENT", np.full(size, np.nan), np.full(size, np.nan), row['L_equipement'], row['m_equipement'], row['L_equipement']/2, '#ADD899'),

    ## ----- PHASE DE CROISIÈRE ----- ##
    ('RÉSERVOIR CROISIÈRE', np.full(size, np.nan), np.full(size, np.nan), row['L_cruise_res'], row['m_cruise_res'], row['L_cruise_res']/2, "#AF47D2"),
    ('PROPERGOL CROISIÈRE', np.full(size, np.nan), np.full(size, np.nan), row['L_cruise_prop'], row['m_cruise_prop'], np.nan, "#AF47D2"),

    ## ----- CASE MOTEUR ----- ##
    ('MOTEUR', np.full(size, np.nan), np.full(size, np.nan), row['L_engine_housing'], row['m_engine_housing'], row['L_engine_housing']/2, '#365E32'),

    ## ----- PHASE ACCÉLÉRATION ----- ##
    ('RÉSERVOIR ACCÉLÉRATION', np.full(size, np.nan), np.full(size, np.nan), row['L_acc_res'], row['m_acc_res'], row['L_acc_res']/2, '#FFC700'),
    ('PROPERGOL ACCÉLÉRATION', np.full(size, np.nan), np.full(size, np.nan), row['L_acc_prop'], row['m_acc_prop'], np.nan, '#FFC700'),

    ## ----- OTHERS ELEMENT ----- ##
    ('TUYÈRE', np.full(size, np.nan), np.full(size, np.nan), row['L_nozzle'], row['m_nozzle'], row['L_nozzle']/2, '#C80036'),
    ("ENTRÉE AIR", np.full(size, np.nan), np.full(size, np.nan), row['L_AirIntakes'], row['m_AirIntakes'], row['L_AirIntakes']/2, '#9BEC00'),
    ("AILES", np.full(size, np.nan), np.full(size, np.nan), row['L_Wings'], row['m_Wings'], row['L_Wings']/2, '#4B70F5'),
    ("QUEUE", np.full(size, np.nan), np.full(size, np.nan), row['L_Tails'], row['m_Tails'], row['L_Tails']/2, '#B1AFFF'),

    ]

    return section_missile