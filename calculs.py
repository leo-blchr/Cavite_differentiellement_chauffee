import equations
import temperature
import numpy as np
import secondmembre


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





### Zone D'execution ###

def main(Grashof, Prandtl):
    

    ### Initialisation
    # Il faut faire en sorte que les valeurs de distances soient données par grashof et prandtl
    matrice_temperature, dx, dy = temperature.maillage_temperature(Lx=5, Ly=5, Nx=10, Ny=10, T_init=0)
    matrice_psi, matrice_omega, _, _ = temperature.maillage_psi_omega(Lx=5, Ly=5, Nx=10, Ny=10, psi_init=0, omega_init=0)


    # Boucle for sur un certain pas de temps qui fait les calculs 

    pas_de_temps = 10
    for i in range (pas_de_temps):

        # Calcul de T au prochain pas de temps 
        T_n_plus_un = calcul_maille_temperature_n_plus_1(T=matrice_temperature, Prandtl=Prandtl, dx=dx, dy=dy, dt=dt, psi=matrice_psi)

        # Calcul d'omega au bord
        omega = calcul_omega_bords(matrice_psi, omega, dx, dy)

        # Calcul d'omega au prochain pas de temps


        # Calcul de psi au prochain pas de temps
    

    pass