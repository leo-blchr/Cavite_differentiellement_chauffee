import algo_thomas
import numpy as np


def calcul(Nx=30, Ny=30, N_iterations=200, dt=0.000005):
    # Paramètres physiques
          # taille de la grille
    Lx, Ly = 1, 1        # longueur physique
    dx = Lx / (Nx-1)
    dy = Ly / (Ny-1)

    nu = 1e-3                # viscosité cinématique réaliste
    Prandt = 0.71            # pour l’air
    beta = 1.0
    g = 9.81
    alpha = 0 *np.pi / 180              # angle du gradient de T
    gamma0 = 1.725             # SOR


    # Initialisation
    T = np.zeros((Nx, Ny))
    omega = np.zeros((Nx, Ny))
    psi = np.zeros((Nx, Ny))

    # Conditions aux limites de T
    T[:, 0] = 1.0   # gauche chaud
    T[:, -1] = 0.0  # droite froid
    T[0, :] = T[1, :]
    T[-1, :] = T[-2, :]

    # Stockage pour visualisation rapide
    liste_T = [T.copy()]
    liste_omega = [omega.copy()]
    liste_psi = [psi.copy()]

    for n in range(N_iterations):
        # ADI température
        T = calcul_maille_temperature_n_plus_1(T, psi, dx, dy, dt, Prandt)

        # ADI vorticité
        #pour le nouveau omega_n+1 on utilise les valeurs de T_n+1 et de psi_n et omega_n
        omega = calcul_maille_omega_n_plus_1(omega, T, psi, dx, dy, dt, nu, beta, g, alpha)

        # SOR pour psi
        for _ in range(100):
            #et psi_n+1 dépend de omega_n+1
            psi = resolution_SOR(psi, omega, gamma0, dx, dy)

        # Sauvegarde
        if n % 10 == 0:  # garder moins de snapshots pour mémoire
            liste_T.append(T.copy())
            liste_omega.append(omega.copy())
            liste_psi.append(psi.copy())

    # Affichage final
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(T, origin='lower', cmap='inferno')
    plt.colorbar()
    plt.title('Température finale')

    plt.subplot(1,2,2)
    plt.imshow(omega, origin='lower', cmap='bwr')
    plt.colorbar()
    plt.title('Vorticité finale')

    plt.show()

    return T, omega, psi, liste_T, liste_omega, liste_psi

# Lancer le test sur 