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

        if i < N_x - 2:
            premier_membre[i-1, i] = dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) - dt/(2*Prandt*dx*dx)
            premier_membre[i, i-1] = -dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) - dt/(2*Prandt*dx*dx)

    return premier_membre, second_membre


        
        
        
        
def calcul_premier_second_membre_2_T(T, phi, j, dx, dy, dt, Prandt):
    (N_x, N_y) = np.shape(T)

    premier_membre = np.zeros((N_y-2, N_y-2))
    second_membre = np.zeros(N_y-2)

    for i in range(1, N_y-1):  # i = 1 → N_y-2
        premier_membre[i-1, i-1] = 1 + dt/(Prandt*dy*dy)

        second_membre[i-1] = (
            T[i, j] * (1 - dt/(Prandt*dx*dx))
            + T[i, j+1] * ( -dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) + dt/(2*Prandt*dx*dx) )
            + T[i, j-1] * (  dt/8*(phi[i, j+1] - phi[i, j-1])/(dx*dy) + dt/(2*Prandt*dx*dx) )
        )

        if i < N_y - 2:
            premier_membre[i-1, i] = -dt/8*(phi[i+1, j] - phi[i-1, j])/(dx*dy) - dt/(2*Prandt*dy*dy)
            premier_membre[i, i-1] =  dt/8*(phi[i+1, j] - phi[i-1, j])/(dx*dy) - dt/(2*Prandt*dy*dy)

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


