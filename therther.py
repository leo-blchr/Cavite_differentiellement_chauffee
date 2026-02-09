import numpy as np
from math import sin, cos 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter



def algorithme_thomas(A, d):
    """
    Résout Ax = d pour une matrice tridiagonale A
    """
    #on va normaliser A et d 
    max_abs_A = np.max(np.abs(A))
    max_abs_d=np.max(np.abs(d))
    max_f=max(max_abs_A,max_abs_d)
    if max_f!=0:
        A=A/max_f
        d=d/max_f

    n = A.shape[0]
    a = np.zeros(n-1)
    b = np.zeros(n)
    c = np.zeros(n-1)

    for i in range(n):
        b[i] = A[i, i]
        if i > 0:
            a[i-1] = A[i, i-1]
        if i < n-1:
            c[i] = A[i, i+1]

    # Forward elimination
    for i in range(1, n):
        m = a[i-1]/b[i-1]
        b[i] -= m*c[i-1]
        d[i] -= m*d[i-1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1]/b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i]*x[i+1])/b[i]

    return x


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
        T_n_plus_demi[1:-1, j] = algorithme_thomas(A_j, b_j)

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
        T_n_plus_un[i, 1:-1] = algorithme_thomas(A_i, b_i)

    # Conditions aux limites en y
    T_n_plus_un[:, 0]  = 1    # bord gauche
    T_n_plus_un[:, -1] = 0    # bord droit

    # Conditions aux limites en x (derivées nulles)
    T_n_plus_un[0, 1:-1]  = T_n_plus_un[1, 1:-1]
    T_n_plus_un[-1, 1:-1] = T_n_plus_un[-2, 1:-1]

    return T_n_plus_un



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
        omega_n_plus_demi[1:-1,j] = algorithme_thomas(A_j, b_j)

    omega_n_plus_un = calcul_omega_bords(psi, dx, dy)

    # ADI direction x
    for i in range(1, Nx-1):
        A_i, b_i = calcul_premier_second_membre_2_omega(
            omega_n_plus_demi, omega_n_plus_un, T_suivant, psi, i, dx, dy, dt, nu, beta, g, alpha
        )
        omega_n_plus_un[i,1:-1] = algorithme_thomas(A_i, b_i)

    return omega_n_plus_un



def resolution_SOR(psi, omega, gamma0, dx, dy):
    Nx, Ny = psi.shape
    beta = dx/dy
    
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            psi_new = (psi[i+1,j] + psi[i-1,j] + beta**2*(psi[i,j+1]+psi[i,j-1]) - dx**2*omega[i,j]) / (2*(1+beta**2))
            # SOR relaxation
            psi[i,j] = (1-gamma0)*psi[i,j] + gamma0*psi_new

    return psi





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


def test_grille_fine(Nx=30, Ny=30, N_iterations=400, dt=0.000005):
    # Paramètres physiques
          # taille de la grille
    Lx, Ly = 1, 1        # longueur physique
    dx = Lx / (Nx-1)
    dy = Ly / (Ny-1)

    nu = 1e-3                # viscosité cinématique réaliste
    Prandt = 0.71            # pour l’air
    beta = 1.0
    g = 9.81
    alpha = 15 *np.pi / 180              # angle du gradient de T
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
