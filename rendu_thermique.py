import numpy as np
from math import sin, cos, pi
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

    premier_membre = np.zeros((Nx, Nx))
    second_membre  = np.zeros(Nx)

    # Condition de Neumann flux nul en i=0
    premier_membre[0, 0] = 1.0
    premier_membre[0, 1] = -1.0
    second_membre[0] = 0.0

    for i in range(1, Nx-1):

        # Vitesses centrées (incompressibilité respectée)
        u = (psi[i, j+1] - psi[i, j-1]) / (2*dy)
        
        v =  -(psi[i+1, j] - psi[i-1, j]) / (2*dx)
        

        # Diagonale principale : diffusion implicite en x
        premier_membre[i, i] = 1 + dt/(Prandt*dx*dx)

        # Second membre : diffusion en y + convection explicite
        second_membre[i] = (
            T[i,j]+
            dt/2*(
                -v*(T[i,j+1]-T[i,j-1])/(2*dy)+
                1/Prandt*(T[i,j+1]-2*T[i,j]+T[i,j-1])/(dy*dy)
                
            )
        )

        premier_membre[i, i+1]   =  dt*u/(4*dx) - dt/(2*Prandt*dx*dx)
        premier_membre[i, i-1] = -dt*u/(4*dx) - dt/(2*Prandt*dx*dx)

    # Condition de Neumann flux nul en i=Nx-1
    premier_membre[-1, -1] = 1.0
    premier_membre[-1, -2] = -1.0
    second_membre[-1] = 0.0

    return premier_membre, second_membre

def calcul_premier_second_membre_2_T(T_un_demi, psi, i, dx, dy, dt, Prandt):
    Nx, Ny = T_un_demi.shape

    premier_membre = np.zeros((Ny-2, Ny-2))
    second_membre  = np.zeros(Ny-2)

    for j in range(1, Ny-1):

        # Vitesses (centrées)
        u = (psi[i, j+1] - psi[i, j-1]) / (2*dy)
        v =  -(psi[i+1, j] - psi[i-1, j]) / (2*dx)
        

        # Diagonale principale (diffusion implicite en y)
        premier_membre[j-1, j-1] = 1 + dt/(Prandt*dy*dy)

        # Second membre : diffusion x + convection explicite
        second_membre[j-1] = (
            T_un_demi[i,j]+
            dt/2*(
                -u*(T_un_demi[i+1,j]-T_un_demi[i-1,j])/(2*dx)+
                1/Prandt*(T_un_demi[i+1,j]-2*T_un_demi[i,j]+T_un_demi[i-1,j])/(dx*dx)
                
            )
        )

        # Points intérieurs
        if 1 < j < Ny-2:
            premier_membre[j-1, j]   =  dt*v/(4*dy) - dt/(2*Prandt*dy*dy)
            premier_membre[j-1, j-2] = -dt*v/(4*dy) - dt/(2*Prandt*dy*dy)

        # Bord gauche (j = 1)
        elif j == 1:
            premier_membre[j-1, j] = dt*v/(4*dy) - dt/(2*Prandt*dy*dy)

            coef = ( -dt*v/(4*dy) - dt/(2*Prandt*dy*dy) )
            second_membre[j-1] -= coef * 1

        # Bord haut (j = Ny-2)
        elif j == Ny-2:
            premier_membre[j-1, j-2] = -dt*v/(4*dy) - dt/(2*Prandt*dy*dy)

            coef = ( dt*v/(4*dy) - dt/(2*Prandt*dy*dy) )
            second_membre[j-1] -= coef * 0

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
        T_n_plus_demi[:, j] = algorithme_thomas(A_j, b_j)

  
    # Parois verticales (x)
    T_n_plus_demi[:, 0]  = 1.0
    T_n_plus_demi[:, -1] = 0.0

    # --------------------
    # Étape 2 : direction x (lignes)
    # --------------------
    for i in range(1, Nx-1):
        A_i, b_i = calcul_premier_second_membre_2_T(T_n_plus_demi, psi, i, dx, dy, dt, Prandt)
        T_n_plus_un[i, 1:-1] = algorithme_thomas(A_i, b_i)

     # Parois verticales (x)
    T_n_plus_un[:, 0]  = 1.0
    T_n_plus_un[:, -1] = 0.0

    # Parois horizontales (y, flux nul après la mise à jour X)
    T_n_plus_un[0, :]  = T_n_plus_un[1, :]
    T_n_plus_un[-1, :] = T_n_plus_un[-2, :]

    return T_n_plus_un


def calcul_omega_bords(psi, dx, dy):
    Nx, Ny = psi.shape
    omega_bords = np.zeros((Nx, Ny))

    # Paroi haute en y (i=0) — paroi horizontale isolée
    for j in range(1, Ny-1):
        omega_bords[0, j] = 2.0*(psi[1,j] - psi[0,j])/dx**2

    # Paroi basse en y (i=Nx-1) — paroi horizontale isolée  
    for j in range(1, Ny-1):
        omega_bords[Nx-1, j] = 2.0*(psi[Nx-2,j] - psi[Nx-1,j])/dx**2

    # Paroi gauche en x (j=0, T=1) — paroi verticale chaude
    for i in range(1, Nx-1):
        omega_bords[i, 0] = 2.0*(psi[i,1] - psi[i,0])/dy**2

    # Paroi droite en x (j=Ny-1, T=0) — paroi verticale froide
    for i in range(1, Nx-1):
        omega_bords[i, Ny-1] = 2.0*(psi[i,Ny-2] - psi[i,Ny-1])/dy**2

    omega_bords[0,0] = omega_bords[0,-1] = 0
    omega_bords[-1,0] = omega_bords[-1,-1] = 0

    return omega_bords


def calcul_premier_second_membre_1_omega(omega, omega_suiv, T_suivant, psi, j, dx, dy, dt,Gr,angle):
    Nx, Ny = omega.shape
    premier_membre = np.zeros((Nx-2, Nx-2))
    second_membre  = np.zeros(Nx-2)

    for i in range(1, Nx-1):
        # Vitesses centrées
        u = (psi[i, j+1] - psi[i, j-1]) / (2*dy)
        v =  -(psi[i+1, j] - psi[i-1, j]) / (2*dx)
        

        # Diagonale principale : diffusion implicite en x
        premier_membre[i-1, i-1] = 1 + dt/(dx*dx)

        # Second membre : diffusion y + convection explicite + terme de flottabilité
        second_membre[i-1] = (
            omega[i,j]+
            dt/2*(
                -v*(omega[i,j+1]-omega[i,j-1])/(2*dy)+
                (omega[i,j+1]-2*omega[i,j]+omega[i,j-1])/(dy*dy)+
                Gr*((T_suivant[i+1,j]-T_suivant[i-1,j])*sin(angle)/(2*dy)+(T_suivant[i,j+1]-T_suivant[i,j-1])*cos(angle)/(2*dx))
            )
        )

        # Points intérieurs
        if 1 < i < Nx-2:
            premier_membre[i-1, i]   =  dt*u/(4*dx) - dt/(2*dx*dx)
            premier_membre[i-1, i-2] = -dt*u/(4*dx) - dt/(2*dx*dx)

        # Bord gauche
        elif i == 1:
            premier_membre[i-1, i] = dt*u/(4*dx) - dt/(2*dx*dx)
            coef = -dt*u/(4*dx) - dt/(2*dx*dx)
            second_membre[i-1] -= coef * omega_suiv[i-1,j]

        # Bord droit
        elif i == Nx-2:
            premier_membre[i-1, i-2] = -dt*u/(4*dx) - dt/(2*dx*dx)
            coef = dt*u/(4*dx) - dt/(2*dx*dx)
            second_membre[i-1] -= coef * omega_suiv[i+1,j]

    return premier_membre, second_membre


def calcul_premier_second_membre_2_omega(omega, omega_suiv, T_suivant, psi, i, dx, dy, dt,Gr,angle):
    Nx, Ny = omega.shape
    premier_membre = np.zeros((Ny-2, Ny-2))
    second_membre  = np.zeros(Ny-2)

    for j in range(1, Ny-1):
        # Vitesses centrées
        u = (psi[i, j+1] - psi[i, j-1]) / (2*dy)
        v =  -(psi[i+1, j] - psi[i-1, j]) / (2*dx)
    

        # Diagonale principale : diffusion implicite en y
        premier_membre[j-1, j-1] = 1 + dt/(dy*dy)

        # Second membre : diffusion x + convection explicite + flottabilité
        second_membre[j-1] = (
            omega[i,j]+
            dt/2*(
                -u*(omega[i+1,j]-omega[i-1,j])/(2*dx)+
                (omega[i+1,j]-2*omega[i,j]+omega[i-1,j])/(dx*dx)+
                Gr*((T_suivant[i+1,j]-T_suivant[i-1,j])*sin(angle)/(2*dy)+(T_suivant[i,j+1]-T_suivant[i,j-1])*cos(angle)/(2*dx))
            )
        )

        # Points intérieurs
        if 1 < j < Ny-2:
            premier_membre[j-1, j]   = dt*v/(4*dy) - dt/(2*dy*dy)
            premier_membre[j-1, j-2] =  -dt*v/(4*dy) - dt/(2*dy*dy)

        # Bord bas
        elif j == 1:
            premier_membre[j-1, j] = dt*v/(4*dy) - dt/(2*dy*dy)
            coef = -dt*v/(4*dy) - dt/(2*dy*dy)
            second_membre[j-1] -= coef * omega_suiv[i,j-1]

        # Bord haut
        elif j == Ny-2:
            premier_membre[j-1, j-2] = -dt*v/(4*dy) - dt/(2*dy*dy)
            coef = dt*v/(4*dy) - dt/(2*dy*dy)
            second_membre[j-1] -= coef * omega_suiv[i,j+1]

    return premier_membre, second_membre


def calcul_maille_omega_n_plus_1(omega, T_suivant, psi, dx, dy, dt,Gr,angle):
    Nx, Ny = omega.shape
    omega_n_plus_demi = calcul_omega_bords(psi, dx, dy)

    # ADI direction y
    for j in range(1, Ny-1):
        A_j, b_j = calcul_premier_second_membre_1_omega(
            omega, omega_n_plus_demi, T_suivant, psi, j, dx, dy, dt,Gr,angle
        )
        omega_n_plus_demi[1:-1,j] = algorithme_thomas(A_j, b_j)

    omega_n_plus_un = calcul_omega_bords(psi, dx, dy)

    # ADI direction x
    for i in range(1, Nx-1):
        A_i, b_i = calcul_premier_second_membre_2_omega(
            omega_n_plus_demi, omega_n_plus_un, T_suivant, psi, i, dx, dy, dt,Gr,angle
        )
        omega_n_plus_un[i,1:-1] = algorithme_thomas(A_i, b_i)

    return omega_n_plus_un



def resolution_SOR(psi, omega, gamma0, dx, dy):
    Nx, Ny = psi.shape
    beta = dx/dy
    
    psi_prec = psi.copy()
    
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            psi_new = (psi[i+1,j] + psi[i-1,j] + beta**2*(psi[i,j+1]+psi[i,j-1]) - dx**2*omega[i,j]) / (2*(1+beta**2))
            # SOR relaxation
            psi[i,j] = (1-gamma0)*psi[i,j] + gamma0*psi_new
            
            
    erreur = np.max(np.abs(psi - psi_prec))
    
    nb_it=2
    
    while erreur > 0.0001:
        psi_prec = psi.copy()
        nb_it=nb_it+1
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                psi_new = (psi[i+1,j] + psi[i-1,j] + beta**2*(psi[i,j+1]+psi[i,j-1]) - dx**2*omega[i,j]) / (2*(1+beta**2))
                # SOR relaxation
                psi[i,j] = (1-gamma0)*psi[i,j] + gamma0*psi_new
                
        erreur = np.max(np.abs(psi - psi_prec))
        
    return psi, nb_it


def condition_arret_temperature(nombre_iteration, T_suiv, T_prec):
    if nombre_iteration <= 10:
        return True
    
    max_difference = np.max(np.abs(T_suiv - T_prec))
    return max_difference > 1e-7




def main(Grashof, Prandtl):

    ### Initialisation
    #nu = 1.5e-5     
    #g = 9.81        
    #beta = 1/300    # coefficient de dilatation thermique [1/K], approximatif pour l'air
    gamma = 1.725
    #nombre_iteration = 100
    dt = 0.000001
    #dt = 0.0001
    angle=0*pi/180

    #Lx = (Grashof * nu**2 / (g * beta * DeltaT))**(1/3)
    Lx = Ly = 1
    Nx = 81
    Ny = 81
    #alpha = nu / Prandtl
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    
    
    # Initialisation
    T = np.zeros((Nx, Ny))
    omega = np.zeros((Nx, Ny))
    psi = np.zeros((Nx, Ny))

    # Conditions aux limites de T
    # Parois verticales
    T[:, 0]  = 1.0   # gauche chaude
    T[:, -1] = 0.0   # droite froide

    # Stockage pour visualisation rapide
    liste_T = [T.copy()]
    liste_omega = [omega.copy()]
    liste_psi = [psi.copy()]
    Tmax = [0]

    T_prec = T
    T_suiv = T
    nombre_iteration = 1  

#    for n in range(nombre_iteration):
    while condition_arret_temperature(nombre_iteration, T_suiv, T_prec):
        # print("passé while")
        if nombre_iteration == 10:
            print(("je suis passé"))
        if nombre_iteration % 500 == 0:
            print(f"Iteration {nombre_iteration} / Diff_T max = {np.max(np.abs(T_suiv - T_prec)):.7e}")
            
        T_prec = T_suiv.copy()
        
        # ADI température
        T = calcul_maille_temperature_n_plus_1(T, psi, dx, dy, dt, Prandtl)
        Tmax.append(T[20, 20])

        # ADI vorticité
        #pour le nouveau omega_n+1 on utilise les valeurs de T_n+1 et de psi_n et omega_n
        omega = calcul_maille_omega_n_plus_1(omega, T, psi, dx, dy, dt,Grashof,angle)

        # SOR pour psi
        
        psi, nb_it = resolution_SOR(psi, omega, gamma, dx, dy)
        #psi = np.zeros((Nx, Ny))


        T_suiv = T
        nombre_iteration += 1

        # print(abs(moy_T_suiv-moy_T_prec)/moy_T_suiv if moy_T_suiv != 0 else 0)
        # print(moy_T_suiv)
        # print(moy_T_prec)
        # Sauvegarde
        if nombre_iteration % 100 == 0:  # garder moins de snapshots pour mémoire
            liste_T.append(T.copy())
            liste_omega.append(omega.copy())
            liste_psi.append(psi.copy())


    print(nombre_iteration, T_suiv, T_prec)
            


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

    plt.figure(figsize=(12,5))
    plt.plot(psi)
    plt.show()
    
    return T, omega, psi, liste_T, liste_omega, liste_psi, Tmax

Ra = 1000000
Prandtl=0.71
Grashof = Ra / Prandtl

# Lancer le test sur une grille fine
T_final, omega_final, psi_final, liste_T, liste_omega, liste_psi, Tmax = main(Grashof, Prandtl)


# -------------------------
# Calcul des vitesses
# -------------------------
def calcul_vitesses(psi, dx, dy):
    Nx, Ny = psi.shape
    u = np.zeros_like(psi)
    w = np.zeros_like(psi)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u[i,j] = (psi[i,j+1] - psi[i,j-1])/(2*dy)
            w[i,j] = -(psi[i+1,j] - psi[i-1,j])/(2*dx)
    return u, w

# -------------------------
# Calcul des extrema
# -------------------------
def calcul_extrema(psi, u, w):
    Nx, Ny = psi.shape
    psi_mid = abs(psi[Nx//2, Ny//2])
    psi_max = np.max(abs(psi))
    pos = np.unravel_index(np.argmax(abs(psi)), psi.shape)
    # u = vitesse horizontale, max sur le plan vertical médian z=0.5 → i = Nx//2
    u_max = np.max(np.abs(u[Nx//2, :]))

    # w = vitesse verticale, max sur le plan horizontal médian x=0.5 → j = Ny//2
    w_max = np.max(np.abs(w[:, Ny//2]))  
    return psi_mid, psi_max, pos, u_max, w_max

# -------------------------
# Calcul de Nusselt
# -------------------------
def calcul_nusselt(T, dx):
    Nx, Ny = T.shape
    Nu = np.zeros(Nx)
    # Paroi gauche (x=0) - Dérivée d'ordre 2 pour plus de précision
    for i in range(Nx):
        Nu[i] = (3*T[i,0] - 4*T[i,1] + T[i,2]) / (2*dx)
        
    Nu_moy = np.trapz(Nu, dx=dx)  # Intégration spatiale trapézoïdale
    Nu_mid = Nu[Nx//2]
    Nu_0 = Nu[0]
    Nu_max = np.max(Nu)
    Nu_min = np.min(Nu)
    Nu_total = np.sum(Nu) * dx
    return Nu_moy, Nu_mid, Nu_0, Nu_max, Nu_min, Nu_total
# -------------------------
# Fonction pour afficher toutes les valeurs
# -------------------------
def afficher_resultats(T, psi, dx, dy, Prandt):
    u, w = calcul_vitesses(psi, dx, dy)
    psi_mid, psi_max, pos, u_max, w_max = calcul_extrema(psi, u, w)
    Nu_moy, Nu_mid, Nu_0, Nu_max, Nu_min, Nu_total = calcul_nusselt(T, dx)

    print(f"|psi_mid| = {psi_mid:.4f}")
    print(f"|psi|max = {psi_max:.4f} position = {pos}")
    print(" (Adimensionnement par le temps thermique : variables * Pr = 0.71)")
    print(f"|psi_mid|      : {psi_mid * Prandt:.3f} ")
    print(f"u_max (horiz)  : {w_max * Prandt:.3f} ")  
    print(f"w_max (vertic) : {u_max * Prandt:.3f} ")
    print(f"Nu_moy         : {Nu_moy:.3f} ")
    print(f"Nu_1/2         : {Nu_mid:.3f} ")
    print(f"Nu_max         : {Nu_max:.3f} ")
    print(f"Nu_min         : {Nu_min:.3f} ")

    return psi_mid, psi_max, pos, u_max, w_max, Nu_moy, Nu_mid, Nu_0, Nu_max, Nu_min, Nu_total

Lx = Ly = 1
Nx = 41
Ny = 41
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
    
psi_mid, psi_max, pos, u_max, w_max, Nu_moy, Nu_mid, Nu_0, Nu_max, Nu_min, Nu_total = afficher_resultats(T_final, psi_final, dx, dy, Prandtl)