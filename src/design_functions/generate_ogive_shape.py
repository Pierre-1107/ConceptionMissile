import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad

def generate_ogive_shape(row, diametre, img_path):
    """
    Génère la forme d'une ogive parabolique, calcule la position du centre de gravité (CG), 
    et affiche un graphique de l'ogive.

    Arguments :
        row : dict
            Dictionnaire contenant les dimensions des sections, notamment la longueur de l'ogive 
            spécifiée par la clé 'L_ogive'.
        diametre : float
            Diamètre maximal de l'ogive.

    Retourne :
        None
            Affiche un graphique représentant l'ogive et son centre de gravité (CG).

    Détails :
        1. **Construction de l'ogive** :
           - L'ogive est modélisée par une équation quadratique de la forme : `y(x) = ax^2 + bx + c`.
           - Les contraintes utilisées pour déterminer les coefficients sont :
               - La courbe commence à (0, 0).
               - La courbe atteint un diamètre maximal de `diametre` à la longueur `row['L_ogive']`.
               - La tangente est horizontale à `row['L_ogive']` (condition de régularité).

        2. **Calcul du centre de gravité (CG)** :
           - La position du CG est calculée comme le rapport des intégrales suivantes :
             \[
             x_{CG} = \frac{\int_0^{L} 2x \cdot y(x) dx}{\int_0^{L} 2 \cdot y(x) dx}
             \]
           - Où \( L \) est la longueur de l'ogive.

        3. **Visualisation** :
           - Le profil de l'ogive est tracé symétriquement autour de l'axe \( x \).
           - Le CG est indiqué par un marqueur rouge sur le graphique.
    """

    x_ogive = np.linspace(0, row['L_ogive'], 1000)

    constraints = [
        (0, 0, False), 
        (row['L_ogive'], diametre/2, False),
        (row['L_ogive'], diametre/2, True),
    ]

    A, B = [], []

    for x_val, y_val, tan_ in constraints:

        if not tan_:

            A.append([x_val**2, x_val, 1])
            B.append(y_val)

        else:

            A.append([2 * x_val, 1, 0])
            B.append(0)

    A = np.array(A)
    B = np.array(B)
    coeffs = np.linalg.solve(A, B)
    a, b, c = coeffs

    def y_ogive(x):
        return a * x**2 + b * x + c

    def integrand_x_area(x):
        return 2 * x * y_ogive(x)

    def integrand_area(x):
        return 2 * y_ogive(x)

    numerator, _ = quad(integrand_x_area, 0, row['L_ogive'])
    denominator, _ = quad(integrand_area, 0, row['L_ogive'])

    x_CG_ogive = numerator / denominator

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    axes.plot(x_ogive, y_ogive(x_ogive), c='black', label=f"y(x) = {a:.3f}x^2 + {b:.3f}x + {c:.3f}")
    axes.plot(x_ogive, -y_ogive(x_ogive), c='black')
    axes.scatter(x_CG_ogive, 0, marker='x', s=100, c='red', label=f"Centre de gravité, x={x_CG_ogive:.5f}")
    axes.grid('on', alpha=0.75, linestyle='-.')
    axes.set_xlabel('Longueur [m]')
    axes.set_ylabel('Diamètre [m]')
    axes.legend(loc="upper left")
    axes.set_title("Profil de l'ogive", fontsize=16)

    fig_path = os.path.join(img_path, "OgiveShape.jpg")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', quality=95)
    plt.show()

    return x_ogive, y_ogive, x_CG_ogive