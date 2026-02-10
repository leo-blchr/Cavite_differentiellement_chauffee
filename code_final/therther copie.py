





import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def sauver_animation(liste_champs, nom_fichier, titre="", cmap="inferno"):
    fig, ax = plt.subplots(figsize=(5,4))

    im = ax.imshow(
        liste_champs[0],
        origin="lower",
        cmap=cmap,
        vmin=np.min(liste_champs),
        vmax=np.max(liste_champs)
    )
    plt.colorbar(im, ax=ax)

    def update(frame):
        im.set_array(liste_champs[frame])
        ax.set_title(f"{titre} – itération {frame}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(liste_champs))

    anim.save(
        nom_fichier,
        writer=PillowWriter(fps=5)
    )

    plt.close()


def test_grille_fine(Nx=30, Ny=30, N_iterations=200, dt=0.000005):
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

# Lancer le test sur une grille fine
T_final, omega_final, psi_final, liste_T, liste_omega, liste_psi = test_grille_fine()


import matplotlib.pyplot as plt
import numpy as np

def plot_cavite_tournee(T, psi, alpha_deg):
    Nx, Ny = T.shape
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Conversion de l'angle en radians
    alpha = np.radians(alpha_deg)

    # Rotation des coordonnées
    X_rot = X*np.cos(alpha) - Y*np.sin(alpha)
    Y_rot = X*np.sin(alpha) + Y*np.cos(alpha)

    plt.figure(figsize=(6,6))
    
    # Champ de température
    plt.contourf(X_rot, Y_rot, T, 20, cmap='inferno')
    plt.colorbar(label='Température')
    
    # Lignes de courant (psi)
    cs = plt.contour(X_rot, Y_rot, psi, colors='white', linewidths=1)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    
    plt.title(f'Cavité penchée alpha = {alpha_deg}°')
    plt.xlabel('X_rot')
    plt.ylabel('Y_rot')
    plt.axis('equal')
    plt.show()

# Exemple avec ton résultat alpha = 15°
plot_cavite_tournee(T_final, psi_final, alpha_deg=15)
