import numpy as np
from math import sin, cos 
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=200)

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


def calcul_premier_second_membre_1_T(T, psi, j, dx, dy, dt, Prandt, debug=False):
    Nx, Ny = T.shape

    premier_membre = np.zeros((Nx-2, Nx-2))
    second_membre  = np.zeros(Nx-2)

    for i in range(1, Nx-1):

        u = (psi[i, j+1] - psi[i, j-1]) / (2*dy)
        v = -(psi[i+1, j] - psi[i-1, j]) / (2*dx)

        # Diagonale principale : diffusion implicite en x
        premier_membre[i-1, i-1] = 1 + dt/(Prandt*dx*dx)

        # Second membre : diffusion en y + convection explicite
        second_membre[i-1] = (
            T[i,j]+
            dt/2*(
                -v*(T[i,j+1]-T[i,j-1])/(2*dy)+
                1/Prandt*(T[i,j+1]-2*T[i,j]+T[i,j-1])/(dy*dy)
                
            )
        )

        # Points intérieurs
        if 1 < i < Nx-2:
            premier_membre[i-1, i]   =  dt*u/(4*dx) - dt/(2*Prandt*dx*dx)
            premier_membre[i-1, i-2] = -dt*u/(4*dx) - dt/(2*Prandt*dx*dx)

        # Bord gauche
        elif i == 1:
            premier_membre[i-1, i] = dt*u/(4*dx) - dt/(2*Prandt*dx*dx)
            premier_membre[i-1, i-1]=premier_membre[i-1, i-1]+ (-dt*u/(4*dx) - dt/(2*Prandt*dx*dx))

        # Bord droit
        elif i == Nx-2:
            premier_membre[i-1, i-2] = -dt*u/(2*dx) - dt/(2*Prandt*dx*dx)
            premier_membre[i-1, i-1]=premier_membre[i-1, i-1]+ (dt*u/(4*dx) - dt/(2*Prandt*dx*dx))

    if debug:
        print(f"\n=== Étape 1 T, colonne j={j} ===")
        print(f"Premier membre (taille {premier_membre.shape}):")
        print(premier_membre)
        print(f"\nSecond membre:")
        print(second_membre)

    return premier_membre, second_membre

def calcul_premier_second_membre_2_T(T, psi, i, dx, dy, dt, Prandt, debug=False):
    Nx, Ny = T.shape

    premier_membre = np.zeros((Ny-2, Ny-2))
    second_membre  = np.zeros(Ny-2)

    for j in range(1, Ny-1):

        u = (psi[i, j+1] - psi[i, j-1]) / (2*dy)
        v = -(psi[i+1, j] - psi[i-1, j]) / (2*dx)

        # Diagonale principale (diffusion implicite en y)
        premier_membre[j-1, j-1] = 1 + dt/(Prandt*dy*dy)

        # Second membre : diffusion x + convection explicite
        second_membre[j-1] = (
            T[i,j]+
            dt/2*(
                -u*(T[i+1,j]-T[i-1,j])/(2*dx)+
                1/Prandt*(T[i+1,j]-2*T[i,j]+T[i-1,j])/(dx*dx)
                
            )
        )

        # Points intérieurs
        if 1 < j < Ny-2:
            premier_membre[j-1, j]   =  dt*v/(4*dy) - dt/(2*Prandt*dy*dy)
            premier_membre[j-1, j-2] = -dt*v/(4*dy) - dt/(2*Prandt*dy*dy)

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

    if debug:
        print(f"\n=== Étape 2 T, ligne i={i} ===")
        print(f"Premier membre (taille {premier_membre.shape}):")
        print(premier_membre)
        print(f"\nSecond membre:")
        print(second_membre)

    return premier_membre, second_membre


def calcul_maille_temperature_n_plus_1(T, psi, dx, dy, dt, Prandt, debug=False):
    Nx, Ny = T.shape
    T_n_plus_demi = np.zeros((Nx, Ny))
    T_n_plus_un   = np.zeros((Nx, Ny))

    if debug:
        print("\n" + "="*80)
        print("CALCUL TEMPÉRATURE")
        print("="*80)
        print(f"\nT initial:")
        print(T)
        print(f"\npsi initial:")
        print(psi)

    # Étape 1 : direction x (colonnes)
    for j in range(1, Ny-1):
        A_j, b_j = calcul_premier_second_membre_1_T(T, psi, j, dx, dy, dt, Prandt, debug=(debug and j==1))
        T_n_plus_demi[1:-1, j] = algorithme_thomas(A_j, b_j)

    # Conditions aux limites en x
    T_n_plus_demi[:, 0]  = 1
    T_n_plus_demi[:, -1] = 0

    # Conditions aux limites en y (derivées nulles)
    T_n_plus_demi[0, 1:-1]  = T_n_plus_demi[1, 1:-1]
    T_n_plus_demi[-1, 1:-1] = T_n_plus_demi[-2, 1:-1]

    if debug:
        print(f"\nT_n+1/2 (après étape 1):")
        print(T_n_plus_demi)

    # Étape 2 : direction y (lignes)
    for i in range(1, Nx-1):
        A_i, b_i = calcul_premier_second_membre_2_T(T_n_plus_demi, psi, i, dx, dy, dt, Prandt, debug=(debug and i==1))
        T_n_plus_un[i, 1:-1] = algorithme_thomas(A_i, b_i)

    # Conditions aux limites en y
    T_n_plus_un[:, 0]  = 1
    T_n_plus_un[:, -1] = 0

    # Conditions aux limites en x (derivées nulles)
    T_n_plus_un[0, 1:-1]  = T_n_plus_un[1, 1:-1]
    T_n_plus_un[-1, 1:-1] = T_n_plus_un[-2, 1:-1]

    if debug:
        print(f"\nT_n+1 (après étape 2, FINAL):")
        print(T_n_plus_un)

    return T_n_plus_un


def calcul_omega_bords(psi, dx, dy):
    Nx, Ny = psi.shape
    omega_bords = np.zeros((Nx, Ny))

    # Paroi basse (j = 0)
    for i in range(1, Nx-1):
        omega_bords[i, 0] = 2.0 * (psi[i, 1] - psi[i, 0]) / dy**2

    # Paroi haute (j = Ny-1)
    for i in range(1, Nx-1):
        omega_bords[i, Ny-1] = 2.0 * (psi[i, Ny-2] - psi[i, Ny-1]) / dy**2

    # Paroi gauche (i = 0)
    for j in range(1, Ny-1):
        omega_bords[0, j] = 2.0 * (psi[1, j] - psi[0, j]) / dx**2

    # Paroi droite (i = Nx-1)
    for j in range(1, Ny-1):
        omega_bords[Nx-1, j] = 2.0 * (psi[Nx-2, j] - psi[Nx-1, j]) / dx**2

    # coins
    omega_bords[0,0] = omega_bords[0,-1] = 0
    omega_bords[-1,0] = omega_bords[-1,-1] = 0

    return omega_bords


def calcul_premier_second_membre_1_omega(omega, omega_suiv, T_suivant, psi, j, dx, dy, dt, alpha, Gr, debug=False):
    Nx, Ny = omega.shape
    premier_membre = np.zeros((Nx-2, Nx-2))
    second_membre  = np.zeros(Nx-2)

    for i in range(1, Nx-1):
        u = (psi[i, j+1] - psi[i, j-1]) / (2*dy)
        v = -(psi[i+1, j] - psi[i-1, j]) / (2*dx)

        # Diagonale principale : diffusion implicite en x
        premier_membre[i-1, i-1] = 1 + dt/(dx*dx)

        # Second membre : diffusion y + convection explicite + terme de flottabilité
        second_membre[i-1] = (
            omega[i,j]+
            dt/2*(
                -v*(omega[i,j+1]-omega[i,j-1])/(2*dy)+
                (omega[i,j+1]-2*omega[i,j]+omega[i,j-1])/(dy*dy)+
                Gr*((T_suivant[i,j+1]-T_suivant[i,j-1])*sin(alpha)/(2*dy)-(T_suivant[i+1,j]-T_suivant[i-1,j])*cos(alpha)/(2*dx))
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

    if debug:
        print(f"\n=== Étape 1 omega, colonne j={j} ===")
        print(f"Premier membre:")
        print(premier_membre)
        print(f"\nSecond membre:")
        print(second_membre)

    return premier_membre, second_membre


def calcul_premier_second_membre_2_omega(omega, omega_suiv, T_suivant, psi, i, dx, dy, dt, alpha, Gr, debug=False):
    Nx, Ny = omega.shape
    premier_membre = np.zeros((Ny-2, Ny-2))
    second_membre  = np.zeros(Ny-2)

    for j in range(1, Ny-1):
        u = (psi[i, j+1] - psi[i, j-1]) / (2*dy)
        v = -(psi[i+1, j] - psi[i-1, j]) / (2*dx)

        # Diagonale principale : diffusion implicite en y
        premier_membre[j-1, j-1] = 1 + dt/(dy*dy)  # CORRIGÉ: dy*dy au lieu de dx*dx

        # Second membre : diffusion x + convection explicite + flottabilité
        second_membre[j-1] = (
            omega[i,j]+
            dt/2*(
                -u*(omega[i+1,j]-omega[i-1,j])/(2*dx)+
                (omega[i+1,j]-2*omega[i,j]+omega[i-1,j])/(dx*dx)+
                Gr*((T_suivant[i,j+1]-T_suivant[i,j-1])*sin(alpha)/(2*dy)-(T_suivant[i+1,j]-T_suivant[i-1,j])*cos(alpha)/(2*dx))
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

    if debug:
        print(f"\n=== Étape 2 omega, ligne i={i} ===")
        print(f"Premier membre:")
        print(premier_membre)
        print(f"\nSecond membre:")
        print(second_membre)

    return premier_membre, second_membre


def calcul_maille_omega_n_plus_1(omega, T_suivant, psi, dx, dy, dt, alpha, Gr, debug=False):
    Nx, Ny = omega.shape
    omega_n_plus_demi = calcul_omega_bords(psi, dx, dy)

    if debug:
        print("\n" + "="*80)
        print("CALCUL VORTICITÉ")
        print("="*80)
        print(f"\nomega initial:")
        print(omega)
        print(f"\nomega_bords (conditions limites):")
        print(omega_n_plus_demi)

    # ADI direction x
    for j in range(1, Ny-1):
        A_j, b_j = calcul_premier_second_membre_1_omega(
            omega, omega_n_plus_demi, T_suivant, psi, j, dx, dy, dt, alpha, Gr, debug=(debug and j==1)
        )
        omega_n_plus_demi[1:-1,j] = algorithme_thomas(A_j, b_j)

    if debug:
        print(f"\nomega_n+1/2 (après étape 1):")
        print(omega_n_plus_demi)

    omega_n_plus_un = calcul_omega_bords(psi, dx, dy)

    # ADI direction y - CORRIGÉ: utiliser i au lieu de j
    for i in range(1, Nx-1):
        A_i, b_i = calcul_premier_second_membre_2_omega(
            omega, omega_n_plus_un, T_suivant, psi, i, dx, dy, dt, alpha, Gr, debug=(debug and i==1))
        omega_n_plus_un[i,1:-1] = algorithme_thomas(A_i, b_i)

    if debug:
        print(f"\nomega_n+1 (après étape 2, FINAL):")
        print(omega_n_plus_un)

    return omega_n_plus_un


def resolution_SOR(psi, omega, gamma0, dx, dy, debug=False):
    Nx, Ny = psi.shape
    beta = dx/dy
    
    if debug:
        print("\n" + "="*80)
        print("RÉSOLUTION SOR pour psi")
        print("="*80)
        print(f"omega utilisé:")
        print(omega)
        print(f"\npsi avant SOR:")
        print(psi)
    
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            psi_new = (psi[i+1,j] + psi[i-1,j] + beta**2*(psi[i,j+1]+psi[i,j-1]) - dx**2*omega[i,j]) / (2*(1+beta**2))
            psi[i,j] = (1-gamma0)*psi[i,j] + gamma0*psi_new

    if debug:
        print(f"\npsi après 1 itération SOR:")
        print(psi)

    return psi


def main_debug():
    ### Paramètres
    Grashof = 1000
    Prandtl = 1
    DeltaT = 20
    
    nu = 1.5e-5     
    g = 9.81        
    beta_therm = 1/300
    gamma = 1.725
    dt = 0.0001  # Pas de temps RÉDUIT

    Lx = (Grashof * nu**2 / (g * beta_therm * DeltaT))**(1/3)
    Ly = Lx
    Nx = 6  # Grille PETITE pour debug
    Ny = 6
    alpha = nu / Prandtl
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    
    print(f"\n{'='*80}")
    print(f"PARAMÈTRES DE LA SIMULATION")
    print(f"{'='*80}")
    print(f"Grashof = {Grashof}")
    print(f"Prandtl = {Prandtl}")
    print(f"dt = {dt}")
    print(f"dx = {dx:.6f}")
    print(f"dy = {dy:.6f}")
    print(f"Lx = Ly = {Lx:.6f}")
    print(f"Nx = Ny = {Nx}")
    print(f"alpha = {alpha:.6e}")
    
    # Initialisation
    T = np.zeros((Nx, Ny))
    omega = np.zeros((Nx, Ny))
    psi = np.zeros((Nx, Ny))

    # Conditions aux limites de T
    T[:, 0] = 1.0   # gauche chaud
    T[:, -1] = 0.0  # droite froid
    T[0, :] = T[1, :]
    T[-1, :] = T[-2, :]

    print(f"\n{'='*80}")
    print(f"ITÉRATION 0 (conditions initiales)")
    print(f"{'='*80}")
    print(f"\nT initial:")
    print(T)
    print(f"\nomega initial:")
    print(omega)
    print(f"\npsi initial:")
    print(psi)

    # UNE SEULE ITÉRATION avec debug complet
    print(f"\n\n{'#'*80}")
    print(f"# ITÉRATION 1 - DÉTAIL COMPLET")
    print(f"{'#'*80}")
    
    # Température
    T = calcul_maille_temperature_n_plus_1(T, psi, dx, dy, dt, Prandtl, debug=True)

    # Vorticité
    omega = calcul_maille_omega_n_plus_1(omega, T, psi, dx, dy, dt, alpha, Grashof, debug=True)

    # SOR (1 seule itération pour debug)
    print("\n" + "="*80)
    print("RÉSOLUTION PSI (1 itération SOR)")
    print("="*80)
    psi = resolution_SOR(psi, omega, gamma, dx, dy, debug=True)

    print(f"\n\n{'='*80}")
    print(f"RÉSULTAT FINAL APRÈS 1 ITÉRATION")
    print(f"{'='*80}")
    print(f"\nT final:")
    print(T)
    print(f"\nomega final:")
    print(omega)
    print(f"\npsi final:")
    print(psi)

    return T, omega, psi, dx, dy, dt, Prandtl, alpha, Grashof

if __name__ == "__main__":
    T, omega, psi, dx, dy, dt, Pr, alpha, Gr = main_debug()