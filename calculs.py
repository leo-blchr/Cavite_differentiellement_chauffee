import equations
import temperature
import numpy as np


def calcul_thomas_ligne_i(T,i):
    second_membre_i = None
    T_i_n_plus_demi = equations.algorithme_thomas(A_i, second_membre_i)
    return T_i_n_plus_demi



def calcul_thomas_colonne_j(T,j):   
    second_membre_j = None
    T_i_n_plus_un = equations.algorithme_thomas(A_j, second_membre_j)
    return T_i_n_plus_un



def calcul_maille_temperature_n_plus_1(T):    
    """
    Calcul de T^{n+1} avec schéma ADI
    """

    Nx, Ny = T.shape
    T_n_plus_demi = np.zeros((Nx, Ny))
    T_n_plus_un = np.zeros((Nx,Ny))

    for j in range(Ny):
        T_n_plus_un[:, j] = calcul_thomas_colonne_j(T_n_plus_demi, j)

    for i in range(Nx):
        T_n_plus_demi[i, :] = calcul_thomas_ligne_i(T, i)
    
    return T_n_plus_un



