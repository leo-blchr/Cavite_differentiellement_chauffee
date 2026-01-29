import equations
import temperature
import numpy as np
import secondmembre
import plots
import os


def calcul_thomas_ligne_i(T,i,second_membre_i,A_i):
    
    T_i_n_plus_demi = equations.algorithme_thomas(A_i, second_membre_i)
    return T_i_n_plus_demi



def calcul_thomas_colonne_j(T,j,second_membre_j,A_j):   
    
    T_i_n_plus_un = equations.algorithme_thomas(A_j, second_membre_j)
    return T_i_n_plus_un



def calcul_maille_temperature_n_plus_1(T,Prandtl,dx,dy,dt,psi):    
    """
    Calcul de T^{n+1} avec schéma ADI
    """
 
    Nx, Ny = T.shape
    T_n_plus_demi = np.zeros((Nx, Ny))
    T_n_plus_un = np.zeros((Nx,Ny))

    for j in range(Ny):
        (A_j,second_membre_j)=secondmembre.calcul_premier_second_membre_1(T,psi,j,dx,dy,dt,Prandtl)
        T_n_plus_demi[:, j] = calcul_thomas_colonne_j(T, j,second_membre_j,A_j)

    for i in range(Nx):
        (A_i,second_membre_i)=secondmembre.calcul_premier_second_membre_2(T,psi,i,dx,dy,dt,Prandtl)
        T_n_plus_un[i, :] = calcul_thomas_ligne_i(T_n_plus_demi, i,second_membre_i,A_i)
    
    return T_n_plus_un



def calcul_omega_bords(psi, omega, dx, dy):
    """
    Calcul de la vorticité aux frontières à partir de psi
    (condition de Thom, parois fixes, non-glissement)
    """
    Nx, Ny = psi.shape

    # --- Paroi basse (j = 0)
    for i in range(1, Nx-1):
        omega[i, 0] = -2.0 * (psi[i, 1] - psi[i, 0]) / dy**2

    # --- Paroi haute (j = Ny-1)
    for i in range(1, Nx-1):
        omega[i, Ny-1] = -2.0 * (psi[i, Ny-2] - psi[i, Ny-1]) / dy**2

    # --- Paroi gauche (i = 0)
    for j in range(1, Ny-1):
        omega[0, j] = -2.0 * (psi[1, j] - psi[0, j]) / dx**2

    # --- Paroi droite (i = Nx-1)
    for j in range(1, Ny-1):
        omega[Nx-1, j] = -2.0 * (psi[Nx-2, j] - psi[Nx-1, j]) / dx**2

    return omega



def calcul_omega_n_plus_1(omega, T, psi, dx, dy, dt, Gr, alpha):

    Nx, Ny = omega.shape
    omega_star = np.zeros_like(omega)
    omega_new  = np.zeros_like(omega)

    # --- demi-pas x implicite
    for j in range(1, Ny-1):
        A, b = secondmembre.calcul_premier_second_membre_1_omega(
            omega, T, psi, j, dx, dy, dt, Gr, alpha
        )
        omega_star[1:Nx-1, j] = equations.algorithme_thomas(A, b)

    # --- demi-pas y implicite
    for i in range(1, Nx-1):
        A, b = secondmembre.calcul_premier_second_membre_2_omega(
            omega_star, T, psi, i, dx, dy, dt, Gr, alpha
        )
        omega_new[i, 1:Ny-1] = equations.algorithme_thomas(A, b)

    return omega_new


def resolution_SOR(psi, omega, gamma, dx, dy):
    Nx, Ny = psi.shape
    beta = dx / dy

    for k in range(2, (Nx-2) + (Ny-2) + 1):
        for i in range(1, Nx-1):
            j = k - i
            if 1 <= j <= Ny-2:

                psi_GS = (psi[i+1, j] + psi[i-1, j] + beta**2 * (psi[i, j+1] + psi[i, j-1]) - dx**2 * omega[i, j]) / (2 * (1 + beta**2))

                psi[i, j] = (1 - gamma) * psi[i, j] + gamma * psi_GS

    return psi

### Ancienne version de SOR codée ###
# A vérifier si la version du dessus est la bonne #

# def resolution_SOR(psi,omega,gamma,dx,dy):
#     Nx, Ny = np.shape(psi)
#     beta = dx / dy
    
#     indices = []

#     for k in range(2, (Nx-2) + (Ny-2) + 1):  # diagonales pour i,j ≥ 1
#         for i in range(1, Nx-1):
#             j = k - i
#             if 1 <= j <= Ny-2:
#                 indices.append((i,j))
                
#     for (i,j) in indices:
#         psi [i,j]= gamma / (2*(1+beta*beta)) * (psi [i+1,j])+psi [i-1,j] + beta * beta * (psi [i,j+1] +psi [i,j-1]) - dx *dx *omega [i, j]
    
#     return(psi)





### Zone D'execution ###

def main(Grashof, Prandtl, save_steps = [0, 2, 5]):
    

    ### Initialisation

    gamma = 1.725
    nombre_iteration = 1000
    dt = 0.01

    # Il faut faire en sorte que les valeurs de distances soient données par grashof et prandtl
    matrice_temperature, dx, dy = temperature.maillage_temperature(Lx=5, Ly=5, Nx=10, Ny=10, T_init=0)
    matrice_psi, matrice_omega, _, _ = temperature.maillage_psi_omega(Lx=5, Ly=5, Nx=10, Ny=10, psi_init=0, omega_init=0)

    # Créer un dossier pour sauvegarder les images si inexistant
    save_dir = "plots_temperature"
    os.makedirs(save_dir, exist_ok=True)


    # Boucle for sur un certain pas de temps qui fait les calculs 
    for i in range (nombre_iteration):

        # Calcul de T au prochain pas de temps 
        T_n_plus_un = calcul_maille_temperature_n_plus_1(T=matrice_temperature, Prandtl=Prandtl, dx=dx, dy=dy, dt=dt, psi=matrice_psi)

        # Calcul d'omega au bord
        matrice_omega = calcul_omega_bords(matrice_psi, matrice_omega, dx, dy)

        # Calcul d'omega au prochain pas de temps
        matrice_omega = calcul_omega_n_plus_1(matrice_omega, T_n_plus_un, matrice_psi, dx, dy, dt, Grashof)

        # Calcul de psi au prochain pas de temps
        matrice_psi = resolution_SOR(matrice_psi, matrice_omega, gamma, dx, dy)

        # --- Sauvegarde du plot si pas de temps concerné
        if i in save_steps:
            filename = os.path.join(save_dir, f"temperature_t{i}.png")
            plots.plot_champ_temperature_brillant(matrice_temperature, titre=f"Champ de T à t={i}", save_filename=filename)
            print(f"Plot sauvegardé : {filename}")

        i += 1

    pass