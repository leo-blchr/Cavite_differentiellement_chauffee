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
    
    T_n_plus_demi_reduit = np.zeros((Nx-2),(Ny-2))
    T_n_plus_un_reduit = np.zeros((Nx-2,Ny-2))
    
    
    
#on calcule le T_n_plus_demi_réduit en enlevant les frontières du domaine
    for j in range(1,Ny-1):
        (A_j,second_membre_j)=secondmembre.calcul_premier_second_membre_1(T,phi,j,dx,dy,dt,Prandt)
        T_n_plus_demi_reduit[:, j-1] = calcul_thomas_colonne_j(T, j,second_membre_j,A_j)

#On reconstruit T_n_plus_demi on respectant les conditions aux limites
    for i in range(0,Nx):
        T_n_plus_demi [i,0]=1
        T_n_plus_demi[i,Ny-1]=0
    
        
    for j in range(0,Ny):
        T_n_plus_demi[0,j]=T_n_plus_demi_reduit[0,j] #dérivées aux limites nulles
        T_n_plus_demi[Nx-1,j]=T_n_plus_demi_reduit[Nx-3,j] #dérivées aux limites nulles 
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            T_n_plus_demi[i,j]=T_n_plus_demi_reduit[i-1,j-1]
            
            
            
# on refait de même avec T_n_plus_un


        

    for i in range(1,Nx-1):
        (A_i,second_membre_i)=secondmembre.calcul_premier_second_membre_2(T,phi,i,dx,dy,dt,Prandt)
        T_n_plus_un_reduit[i-1, :] = calcul_thomas_ligne_i(T_n_plus_demi, i,second_membre_i,A_i)
        
    #On reconstruit T_n_plus_un on respectant les conditions aux limites
    for i in range(0,Nx):
        T_n_plus_un [i,0]=1
        T_n_plus_un[i,Ny-1]=0
    
        
    for j in range(0,Ny):
        T_n_plus_un[0,j]=T_n_plus_un_reduit[0,j] #dérivées aux limites nulles
        T_n_plus_demi[Nx-1,j]=T_n_plus_un_reduit[Nx-3,j] #dérivées aux limites nulles 
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            T_n_plus_un[i,j]=T_n_plus_un_reduit[i-1,j-1]
    
    return T_n_plus_un



