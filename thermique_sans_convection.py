import numpy as np

# --------------------
# Algorithme de Thomas pour tridiagonale
# --------------------
def algorithme_thomas(A, d):
    n = len(d)
    a = np.zeros(n-1)
    b = np.zeros(n)
    c = np.zeros(n-1)
    for i in range(n):
        b[i] = A[i,i]
        if i > 0:
            a[i-1] = A[i,i-1]
        if i < n-1:
            c[i] = A[i,i+1]

    # Forward elimination
    for i in range(1,n):
        m = a[i-1]/b[i-1]
        b[i] -= m*c[i-1]
        d[i] -= m*d[i-1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1]/b[-1]
    for i in range(n-2,-1,-1):
        x[i] = (d[i]-c[i]*x[i+1])/b[i]

    return x

# --------------------
# ADI Temperature
# --------------------
def calcul_maille_temperature_n_plus_1(T, psi, dx, dy, dt, Pr):
    Nx, Ny = T.shape
    T_half = T.copy()
    T_new = T.copy()

    # Direction y
    for j in range(1, Ny-1):
        A = np.zeros((Nx-2,Nx-2))
        b = np.zeros(Nx-2)
        for i in range(1,Nx-1):
            ip = min(i+1,Nx-1)
            im = max(i-1,0)

            A[i-1,i-1] = 1 + dt/(Pr*dx**2)
            if i-2 >=0:
                A[i-1,i-2] = -dt/(2*Pr*dx**2)
            if i < Nx-2:
                A[i-1,i] = -dt/(2*Pr*dx**2)

            b[i-1] = T[i,j]*(1 - dt/(Pr*dy**2)) + dt/(2*Pr*dy**2)*(T[i,j+1]+T[i,j-1])

        T_half[1:-1,j] = algorithme_thomas(A,b)

    # Conditions aux limites
    T_half[:,0] = 1.0
    T_half[:,-1] = 0.0
    T_half[0,1:-1] = T_half[1,1:-1]
    T_half[-1,1:-1] = T_half[-2,1:-1]

    # Direction x
    for i in range(1,Nx-1):
        A = np.zeros((Ny-2,Ny-2))
        b = np.zeros(Ny-2)
        for j in range(1,Ny-1):
            jp = min(j+1,Ny-1)
            jm = max(j-1,0)

            A[j-1,j-1] = 1 + dt/(Pr*dy**2)
            if j-2 >=0:
                A[j-1,j-2] = -dt/(2*Pr*dy**2)
            if j < Ny-2:
                A[j-1,j] = -dt/(2*Pr*dy**2)

            b[j-1] = T_half[i,j]*(1 - dt/(Pr*dx**2)) + dt/(2*Pr*dx**2)*(T_half[i+1,j]+T_half[i-1,j])

        T_new[i,1:-1] = algorithme_thomas(A,b)

    # Conditions aux limites
    T_new[:,0] = 1.0
    T_new[:,-1] = 0.0
    T_new[0,1:-1] = T_new[1,1:-1]
    T_new[-1,1:-1] = T_new[-2,1:-1]

    return T_new

# --------------------
# SOR pour psi
# --------------------
def resolution_SOR(psi, omega, gamma, dx, dy, Niter=50):
    Nx, Ny = psi.shape
    beta = dx/dy
    for _ in range(Niter):
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                psi_new = (psi[i+1,j] + psi[i-1,j] + beta**2*(psi[i,j+1]+psi[i,j-1]) - dx**2*omega[i,j])/(2*(1+beta**2))
                psi[i,j] = (1-gamma)*psi[i,j] + gamma*psi_new
    return psi

# --------------------
# Vorticité: bords
# --------------------
def calcul_omega_bords(psi, dx, dy):
    Nx, Ny = psi.shape
    omega = np.zeros((Nx,Ny))
    omega[1:-1,0] = -2*(psi[1:-1,1]-psi[1:-1,0])/dx**2
    omega[1:-1,-1] = -2*(psi[1:-1,-2]-psi[1:-1,-1])/dx**2
    omega[0,1:-1] = -2*(psi[1,1:-1]-psi[0,1:-1])/dy**2
    omega[-1,1:-1] = -2*(psi[-2,1:-1]-psi[-1,1:-1])/dy**2
    return omega

# --------------------
# ADI Vorticité
# --------------------
def calcul_maille_omega_n_plus_1(omega, T, psi, dx, dy, dt, nu, beta_g, g, alpha):
    Nx, Ny = omega.shape
    omega_half = calcul_omega_bords(psi, dx, dy)
    omega_new = omega_half.copy()

    # --------------------
    # direction y (colonnes)
    # --------------------
    for j in range(1, Ny-1):
        A = np.zeros((Nx-2, Nx-2))
        b = np.zeros(Nx-2)
        jp = min(j+1, Ny-1)
        jm = max(j-1, 0)

        for i in range(1, Nx-1):
            ip = min(i+1, Nx-1)
            im = max(i-1, 0)

            # Tridiagonale
            A[i-1, i-1] = 1 + dt*nu/(dx**2)
            if i-2 >= 0:
                A[i-1, i-2] = -dt*nu/(2*dx**2)
            if i < Nx-2:
                A[i-1, i] = -dt*nu/(2*dx**2)

            # Second membre : diffusion + T forcing
            b[i-1] = omega[i,j]*(1 - dt*nu/(dy**2)) \
                     + dt*nu/(2*dy**2)*(omega[i,jp]+omega[i,jm]) \
                     + beta_g*g*((T[i,jp]-T[i,jm])*np.sin(alpha)/dy - (T[ip,j]-T[im,j])*np.cos(alpha)/dx)

        omega_half[1:-1, j] = algorithme_thomas(A, b)

    # --------------------
    # direction x (lignes)
    # --------------------
    for i in range(1, Nx-1):
        A = np.zeros((Ny-2, Ny-2))
        b = np.zeros(Ny-2)
        ip = min(i+1, Nx-1)
        im = max(i-1, 0)

        for j in range(1, Ny-1):
            jp = min(j+1, Ny-1)
            jm = max(j-1, 0)

            A[j-1, j-1] = 1 + dt*nu/(dy**2)
            if j-2 >= 0:
                A[j-1, j-2] = -dt*nu/(2*dy**2)
            if j < Ny-2:
                A[j-1, j] = -dt*nu/(2*dy**2)

            b[j-1] = omega_half[i,j]*(1 - dt*nu/(dx**2)) \
                     + dt*nu/(2*dx**2)*(omega_half[ip,j]+omega_half[im,j]) \
                     + beta_g*g*((T[i,jp]-T[i,jm])*np.sin(alpha)/dy - (T[ip,j]-T[im,j])*np.cos(alpha)/dx)

        omega_new[i, 1:-1] = algorithme_thomas(A, b)

    return omega_new


# --------------------
# Exemple d'utilisation sur grille fine
# --------------------
Nx, Ny = 50, 50
dx = dy = 1.0/(Nx-1)
dt = 0.001
nu = 1e-3
Pr = 0.71
beta_g = 1.0
g = 9.81
alpha = 0.0
gamma = 1.5

T = np.zeros((Nx,Ny))
T[:,0] = 1.0
T[:,-1] = 0.0
T[0,:] = T[1,:]
T[-1,:] = T[-2,:]

omega = np.zeros((Nx,Ny))
psi = np.zeros((Nx,Ny))

N_iter = 100
for n in range(N_iter):
    T = calcul_maille_temperature_n_plus_1(T, psi, dx, dy, dt, Pr)
    omega = calcul_maille_omega_n_plus_1(omega, T, psi, dx, dy, dt, nu, beta_g, g, alpha)
    psi = resolution_SOR(psi, omega, gamma, dx, dy, Niter=20)

print("Température finale :")
print(T)


import matplotlib.pyplot as plt

def affiche_champs(T, omega, psi, titre_prefix="Itération"):
    """
    Affiche T, omega et psi côte à côte.
    """
    plt.figure(figsize=(15,4))

    # Température
    plt.subplot(1,3,1)
    plt.imshow(T, origin="lower", cmap="inferno")
    plt.colorbar()
    plt.title(f"{titre_prefix} - Température")

    # Vorticité
    plt.subplot(1,3,2)
    plt.imshow(omega, origin="lower", cmap="bwr")
    plt.colorbar()
    plt.title(f"{titre_prefix} - Vorticité")

    # Psi
    plt.subplot(1,3,3)
    plt.imshow(psi, origin="lower", cmap="viridis")
    plt.colorbar()
    plt.title(f"{titre_prefix} - Psi")

    plt.tight_layout()
    plt.show()


for n in range(N_iter):
    T = calcul_maille_temperature_n_plus_1(T, psi, dx, dy, dt, Pr)
    omega = calcul_maille_omega_n_plus_1(omega, T, psi, dx, dy, dt, nu, beta_g, g, alpha)
    psi = resolution_SOR(psi, omega, gamma, dx, dy, Niter=20)

    if n % 10 == 0:
        affiche_champs(T, omega, psi, titre_prefix=f"Itération {n}")