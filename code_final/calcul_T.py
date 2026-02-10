import algo_thomas
import numpy as np

def calcul_premier_second_membre_1_T(T, psi, j, dx, dy, dt, Prandt):
    Nx, Ny = T.shape

    premier_membre = np.zeros((Nx-2, Nx-2))
    second_membre  = np.zeros(Nx-2)

    for i in range(1, Nx-1):

        # Vitesses centrées (incompressibilité respectée)
        u = -(psi[i, j+1] - psi[i, j-1]) / (2*dy)
        v =  (psi[i+1, j] - psi[i-1, j]) / (2*dx)

        # Diagonale principale : diffusion implicite en x
        premier_membre[i-1, i-1] = 1 + dt/(Prandt*dx*dx)

        # Second membre : diffusion en y + convection explicite
        second_membre[i-1] = (
            T[i, j] * (1 - dt/(Prandt*dy*dy))
            + T[i, j+1] * ( -dt*v/(2*dy) + dt/(2*Prandt*dy*dy) )
            + T[i, j-1] * (  dt*v/(2*dy) + dt/(2*Prandt*dy*dy) )
        )

        # Points intérieurs
        if 1 < i < Nx-2:
            premier_membre[i-1, i]   =  dt*u/(2*dx) - dt/(2*Prandt*dx*dx)
            premier_membre[i-1, i-2] = -dt*u/(2*dx) - dt/(2*Prandt*dx*dx)

        # Bord gauche
        elif i == 1:
            premier_membre[i-1, i] = dt*u/(2*dx) - dt/(2*Prandt*dx*dx)

            coef = (-dt*u/(2*dx) - dt/(2*Prandt*dx*dx))
            second_membre[i-1] -= coef * T[i-1, j]

        # Bord droit
        elif i == Nx-2:
            premier_membre[i-1, i-2] = -dt*u/(2*dx) - dt/(2*Prandt*dx*dx)

            coef = ( dt*u/(2*dx) - dt/(2*Prandt*dx*dx))
            second_membre[i-1] -= coef * T[i+1, j]

    return premier_membre, second_membre

def calcul_premier_second_membre_2_T(T, psi, i, dx, dy, dt, Prandt):
    Nx, Ny = T.shape

    premier_membre = np.zeros((Ny-2, Ny-2))
    second_membre  = np.zeros(Ny-2)

    for j in range(1, Ny-1):

        # Vitesses (centrées)
        u = -(psi[i, j+1] - psi[i, j-1]) / (2*dy)
        v =  (psi[i+1, j] - psi[i-1, j]) / (2*dx)

        # Diagonale principale (diffusion implicite en y)
        premier_membre[j-1, j-1] = 1 + dt/(Prandt*dy*dy)

        # Second membre : diffusion x + convection explicite
        second_membre[j-1] = (
            T[i, j] * (1 - dt/(Prandt*dx*dx))
            + T[i+1, j] * ( -dt*u/(4*dx) + dt/(2*Prandt*dx*dx) )
            + T[i-1, j] * (  dt*u/(4*dx) + dt/(2*Prandt*dx*dx) )
        )

        # Points intérieurs
        if 1 < j < Ny-2:
            premier_membre[j-1, j]   = -dt*v/(4*dy) - dt/(2*Prandt*dy*dy)
            premier_membre[j-1, j-2] =  dt*v/(4*dy) - dt/(2*Prandt*dy*dy)

        # Bord bas (j = 1)
        elif j == 1:
            premier_membre[j-1, j] = -dt*v/(4*dy) - dt/(2*Prandt*dy*dy)

            coef = ( dt*v/(4*dy) - dt/(2*Prandt*dy*dy) )
            second_membre[j-1] -= coef * T[i, j-1]

        # Bord haut (j = Ny-2)
        elif j == Ny-2:
            premier_membre[j-1, j-2] = dt*v/(4*dy) - dt/(2*Prandt*dy*dy)

            coef = ( -dt*v/(4*dy) - dt/(2*Prandt*dy*dy) )
            second_membre[j-1] -= coef * T[i, j+1]

    return premier_membre, second_membre


def calcul_maille_temperature_n_plus_1(T, psi, dx, dy, dt, Prandt):
    Nx, Ny = T.shape
    T_n_plus_demi = np.zeros((Nx, Ny))
    T_n_plus_un   = np.zeros((Nx, Ny))

    # --------------------
    # Étape 1 : direction y (colonnes)
    # --------------------
    for j in range(1, Ny-1):
        A_j, b_j = calcul_premier_second_membre_1_T(T, psi, j, dx, dy, dt, Prandt)
        T_n_plus_demi[1:-1, j] = algo_thomas.algorithme_thomas(A_j, b_j)

    # Conditions aux limites en x
    T_n_plus_demi[:, 0]  = 1    # bord gauche
    T_n_plus_demi[:, -1] = 0    # bord droit

    # Conditions aux limites en y (derivées nulles)
    T_n_plus_demi[0, 1:-1]  = T_n_plus_demi[1, 1:-1]
    T_n_plus_demi[-1, 1:-1] = T_n_plus_demi[-2, 1:-1]

    # --------------------
    # Étape 2 : direction x (lignes)
    # --------------------
    for i in range(1, Nx-1):
        A_i, b_i = calcul_premier_second_membre_2_T(T_n_plus_demi, psi, i, dx, dy, dt, Prandt)
        T_n_plus_un[i, 1:-1] = algo_thomas.algorithme_thomas(A_i, b_i)

    # Conditions aux limites en y
    T_n_plus_un[:, 0]  = 1    # bord gauche
    T_n_plus_un[:, -1] = 0    # bord droit

    # Conditions aux limites en x (derivées nulles)
    T_n_plus_un[0, 1:-1]  = T_n_plus_un[1, 1:-1]
    T_n_plus_un[-1, 1:-1] = T_n_plus_un[-2, 1:-1]

    return T_n_plus_un