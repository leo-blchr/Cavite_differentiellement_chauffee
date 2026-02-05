import equations
import numpy as np
from math import sin, cos 

def calcul_premier_second_membre_1_T(T, phi, j, dx, dy, dt, Prandt):
    (N_x, N_y) = np.shape(T)

    premier_membre = np.zeros((N_x-2, N_x-2))
    second_membre = np.zeros(N_x-2)

    for i in range(1, N_x-1):  # i = 1 → N_x-2
        premier_membre[i-1, i-1] = 1 + dt/(Prandt*dx*dx)

        second_membre[i-1] = (
            T[i, j] * (1 - dt/(Prandt*dy*dy))
            + T[i, j+1] * ( dt/8*(phi[i+1, j] - phi[i-1, j])/(dx*dy) + dt/(2*Prandt*dy*dy) )
            + T[i, j-1] * ( -dt/8*(phi[i+1, j] - phi[i-1, j])/(dx*dy) + dt/(2*Prandt*dy*dy) )
        )

        if 1< i < N_x - 2:
            premier_membre[i-1, i] = dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) - dt/(2*Prandt*dx*dx)
            premier_membre[i-1, i-2] = -dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) - dt/(2*Prandt*dx*dx)
        
        elif i==1:
            premier_membre[i-1, i] = dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) - dt/(2*Prandt*dx*dx)
            premier_membre[i-1, i-1] = premier_membre [i-1, i-1] -dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) - dt/(2*Prandt*dx*dx)
            
        elif i ==N_x-2:
            premier_membre[i-1, i-2] = -dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) - dt/(2*Prandt*dx*dx)
            premier_membre[i-1, i-1] = premier_membre [i-1, i-1]+dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) - dt/(2*Prandt*dx*dx)
            
            

    return premier_membre, second_membre


        
        
        
        
def calcul_premier_second_membre_2_T(T, phi, i, dx, dy, dt, Prandt):
    (N_x, N_y) = np.shape(T)

    premier_membre = np.zeros((N_y-2, N_y-2))
    second_membre = np.zeros(N_y-2)

    for j in range(1, N_y-1):  # i = 1 → N_y-2
        premier_membre[j-1, j-1] = 1 + dt/(Prandt*dy*dy)

        second_membre[j-1] = (
            T[i, j] * (1 - dt/(Prandt*dx*dx))
            + T[i+1, j] * ( -dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) + dt/(2*Prandt*dx*dx) )
            + T[i-1, j] * (  dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) + dt/(2*Prandt*dx*dx) )
        )
        
        #Dans le cas ou j= N_y-2, on est Ti,Ny-1= 0 donc le terme en Ti,j+1 s'annule par contre dans le cas en j=1 le terme en Ti,j-1 vaut
        # 1 donc il faut ajouter un terme dans le second membre 
         
        
        if 1< j < N_y - 2:
            premier_membre[j-1, j] = -dt/8*(phi[i+1, j] - phi[i-1, j])/(dx*dy) - dt/(2*Prandt*dy*dy)
            premier_membre[j-1, j-2] =  dt/8*(phi[i+1, j] - phi[i-1, j])/(dx*dy) - dt/(2*Prandt*dy*dy)
        
        elif j==1:
            premier_membre[j-1, j] = -dt/8*(phi[i+1, j] - phi[i-1, j])/(dx*dy) - dt/(2*Prandt*dy*dy)
            second_membre[j-1]= second_membre[j-1]- 1 *(dt/8*(phi[i+1, j] - phi[i-1, j])/(dx*dy) - dt/(2*Prandt*dy*dy))
            
            
        elif j ==N_y-2:
           premier_membre[j-1,j-2]=dt/8*(phi[i+1, j] - phi[i-1, j])/(dx*dy) - dt/(2*Prandt*dy*dy)
           second_membre[j-1]= second_membre[j-1]- 0 *(phi[i+1, j] - phi[i-1, j])/(dx*dy) - dt/(2*Prandt*dy*dy)
           
        


    return premier_membre, second_membre

                
                    


def calcul_premier_second_membre_1_omega(omega, T_suivant, phi, j, dx, dy, dt, nu,beta,g,alpha):
    (N_x, N_y) = np.shape(omega)

    premier_membre = np.zeros((N_x-2, N_x-2))
    second_membre = np.zeros(N_x-2)

    for i in range(1, N_x-1):   # i = 1 ... N_x-2

        premier_membre[i-1, i-1] = 1 + dt * nu / (dx*dx)

        second_membre[i-1] = (
            omega[i, j] * (1 - dt * nu / (dy*dy))
            + omega[i, j+1] * (dt/8 * (phi[i+1, j] - phi[i-1, j]) / (dx*dy) + dt * nu / (2*dy*dy))
            + omega[i, j-1] * (-dt/8 * (phi[i+1, j] - phi[i-1, j]) / (dx*dy) + dt * nu / (2*dy*dy))
            + beta*g* (T_suivant[i, j+1] - T_suivant[i,j-1]) * sin(alpha)/dy -(T_suivant[i+1, j]-T_suivant[i-1,j]) * cos(alpha)/dx
        )
        

        if i < N_x - 2:
            premier_membre[i-1, i] = dt/8 * (phi[i, j+1] - phi[i, j-1])/(dx*dy) - dt * nu/(2*dx*dx)
            premier_membre[i, i-1] = -dt/8 * (phi[i, j+1] - phi[i, j-1])/(dx*dy) - dt * nu/(2*dx*dx)

    return premier_membre, second_membre

        
        
        
def calcul_premier_second_membre_2_omega(omega, T_suivant, phi, j, dx, dy, dt, nu,beta,g,alpha):
    (N_x, N_y) = np.shape(omega)

    premier_membre = np.zeros((N_y-2, N_y-2))
    second_membre = np.zeros(N_y-2)

    for i in range(1, N_y-1):   # i = 1 ... N_y-2

        premier_membre[i-1, i-1] = 1 + dt * nu / (dy*dy)

        second_membre[i-1] = (
            omega[i, j] * (1 - dt * nu / (dx*dx))
            + omega[i, j+1] * (-dt/8 * (phi[i, j+1] - phi[i, j-1])/(dx*dy) + dt * nu / (2*dx*dx))
            + omega[i, j-1] * ( dt/8 * (phi[i, j+1] - phi[i, j-1])/(dx*dy) + dt * nu / (2*dx*dx))
            + beta*g* (T_suivant[i, j+1] - T_suivant[i,j-1]) * sin(alpha)/dy -(T_suivant[i+1, j]-T_suivant[i-1,j]) * cos(alpha)/dx
        )

        if i < N_y - 2:
            premier_membre[i-1, i] = -dt/8 * (phi[i+1, j] - phi[i-1, j])/(dx*dy) - dt * nu/(2*dy*dy)
            premier_membre[i, i-1] =  dt/8 * (phi[i+1, j] - phi[i-1, j])/(dx*dy) - dt * nu/(2*dy*dy)

    return premier_membre, second_membre


