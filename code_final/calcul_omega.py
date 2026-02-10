import algo_thomas
import numpy as np
def calcul_omega_bords(psi, dx, dy):
    Nx, Ny = psi.shape
    omega_bords = np.zeros((Nx, Ny))

     # --- Paroi basse (j = 0)
    for i in range(1, Nx-1):
        omega_bords[i, 0] = -2.0 * (psi[i, 1] - psi[i, 0]) / dy**2

    # --- Paroi haute (j = Ny-1)
    for i in range(1, Nx-1):
        omega_bords[i, Ny-1] = -2.0 * (psi[i, Ny-2] - psi[i, Ny-1]) / dy**2

    # --- Paroi gauche (i = 0)
    for j in range(1, Ny-1):
        omega_bords[0, j] = -2.0 * (psi[1, j] - psi[0, j]) / dx**2

    # --- Paroi droite (i = Nx-1)
    for j in range(1, Ny-1):
        omega_bords[Nx-1, j] = -2.0 * (psi[Nx-2, j] - psi[Nx-1, j]) / dx**2

    # coins
    omega_bords[0,0] = omega_bords[0,-1] = 0
    omega_bords[-1,0] = omega_bords[-1,-1] = 0

    return omega_bords


def calcul_premier_second_membre_1_omega(omega, omega_suiv, T_suivant, psi, j, dx, dy, dt, nu, beta, g, alpha):
    Nx, Ny = omega.shape
    premier_membre = np.zeros((Nx-2, Nx-2))
    second_membre  = np.zeros(Nx-2)

    for i in range(1, Nx-1):
        # Vitesses centrées
        u = -(psi[i, j+1] - psi[i, j-1]) / (2*dy)
        v =  (psi[i+1, j] - psi[i-1, j]) / (2*dx)

        # Diagonale principale : diffusion implicite en x
        premier_membre[i-1, i-1] = 1 + dt*nu/(dx*dx)

        # Second membre : diffusion y + convection explicite + terme de flottabilité
        second_membre[i-1] = (
            omega[i,j]*(1 - dt*nu/(dy*dy)) +
            omega[i,j+1]*( -dt*v/(2*dy) + dt*nu/(2*dy*dy) ) +
            omega[i,j-1]*(  dt*v/(2*dy) + dt*nu/(2*dy*dy) ) +
            beta*g*( (T_suivant[i,j+1]-T_suivant[i,j-1])*np.sin(alpha)/dy -
                     (T_suivant[i+1,j]-T_suivant[i-1,j])*np.cos(alpha)/dx )
        )

        # Points intérieurs
        if 1 < i < Nx-2:
            premier_membre[i-1, i]   =  dt*u/(2*dx) - dt*nu/(2*dx*dx)
            premier_membre[i-1, i-2] = -dt*u/(2*dx) - dt*nu/(2*dx*dx)

        # Bord gauche
        elif i == 1:
            premier_membre[i-1, i] = dt*u/(2*dx) - dt*nu/(2*dx*dx)
            coef = -dt*u/(2*dx) - dt*nu/(2*dx*dx)
            second_membre[i-1] -= coef * omega_suiv[i-1,j]

        # Bord droit
        elif i == Nx-2:
            premier_membre[i-1, i-2] = -dt*u/(2*dx) - dt*nu/(2*dx*dx)
            coef = dt*u/(2*dx) - dt*nu/(2*dx*dx)
            second_membre[i-1] -= coef * omega_suiv[i+1,j]

    return premier_membre, second_membre


def calcul_premier_second_membre_2_omega(omega, omega_suiv, T_suivant, psi, i, dx, dy, dt, nu, beta, g, alpha):
    Nx, Ny = omega.shape
    premier_membre = np.zeros((Ny-2, Ny-2))
    second_membre  = np.zeros(Ny-2)

    for j in range(1, Ny-1):
        # Vitesses centrées
        u = -(psi[i, j+1] - psi[i, j-1]) / (2*dy)
        v =  (psi[i+1, j] - psi[i-1, j]) / (2*dx)

        # Diagonale principale : diffusion implicite en y
        premier_membre[j-1, j-1] = 1 + dt*nu/(dy*dy)

        # Second membre : diffusion x + convection explicite + flottabilité
        second_membre[j-1] = (
            omega[i,j]*(1 - dt*nu/(dx*dx)) +
            omega[i+1,j]*( -dt*u/(2*dx) + dt*nu/(2*dx*dx) ) +
            omega[i-1,j]*(  dt*u/(2*dx) + dt*nu/(2*dx*dx) ) +
            beta*g*( (T_suivant[i,j+1]-T_suivant[i,j-1])*np.sin(alpha)/dy -
                     (T_suivant[i+1,j]-T_suivant[i-1,j])*np.cos(alpha)/dx )
        )

        # Points intérieurs
        if 1 < j < Ny-2:
            premier_membre[j-1, j]   = -dt*v/(2*dy) - dt*nu/(2*dy*dy)
            premier_membre[j-1, j-2] =  dt*v/(2*dy) - dt*nu/(2*dy*dy)

        # Bord bas
        elif j == 1:
            premier_membre[j-1, j] = -dt*v/(2*dy) - dt*nu/(2*dy*dy)
            coef = dt*v/(2*dy) - dt*nu/(2*dy*dy)
            second_membre[j-1] -= coef * omega_suiv[i,j-1]

        # Bord haut
        elif j == Ny-2:
            premier_membre[j-1, j-2] = dt*v/(2*dy) - dt*nu/(2*dy*dy)
            coef = -dt*v/(2*dy) - dt*nu/(2*dy*dy)
            second_membre[j-1] -= coef * omega_suiv[i,j+1]

    return premier_membre, second_membre


def calcul_maille_omega_n_plus_1(omega, T_suivant, psi, dx, dy, dt, nu, beta, g, alpha):
    Nx, Ny = omega.shape
    omega_n_plus_demi = calcul_omega_bords(psi, dx, dy)

    # ADI direction y
    for j in range(1, Ny-1):
        A_j, b_j = calcul_premier_second_membre_1_omega(
            omega, omega_n_plus_demi, T_suivant, psi, j, dx, dy, dt, nu, beta, g, alpha
        )
        omega_n_plus_demi[1:-1,j] = algo_thomas.algorithme_thomas(A_j, b_j)

    omega_n_plus_un = calcul_omega_bords(psi, dx, dy)

    # ADI direction x
    for i in range(1, Nx-1):
        A_i, b_i = calcul_premier_second_membre_2_omega(
            omega_n_plus_demi, omega_n_plus_un, T_suivant, psi, i, dx, dy, dt, nu, beta, g, alpha
        )
        omega_n_plus_un[i,1:-1] = algo_thomas.algorithme_thomas(A_i, b_i)

    return omega_n_plus_un


