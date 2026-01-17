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



def calcul_maille_temperature_n_plus_1(T,Prandt,dx,dy,dt,phi):    
    """
    Calcul de T^{n+1} avec schéma ADI
    """
 
    Nx, Ny = T.shape
    T_n_plus_demi = np.zeros((Nx, Ny))
    T_n_plus_un = np.zeros((Nx,Ny))

    for j in range(Ny):
        (A_j,second_membre_j)=secondmembre.calcul_premier_second_membre_1(T,phi,j,dx,dy,dt,Prandt)
        T_n_plus_demi[:, j] = calcul_thomas_colonne_j(T, j,second_membre_j,A_j)

    for i in range(Nx):
        (A_i,second_membre_i)=secondmembre.calcul_premier_second_membre_2(T,phi,i,dx,dy,dt,Prandt)
        T_n_plus_un[i, :] = calcul_thomas_ligne_i(T_n_plus_demi, i,second_membre_i,A_i)
    
    return T_n_plus_un



