import equations
import numpy as np



def maillage_temperature(Lx, Ly, Nx, Ny, T_init):
    """
    Crée un maillage 2D de température sur [0,Lx]x[0,Ly]
    
    Returns:
        x, y : grilles 2D
        T    : champ de température (Ny, Nx)
        dx, dy : pas du maillage
    """

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    X, Y = np.meshgrid(x, y, indexing="ij")

    T = T_init * np.ones((Nx, Ny))

    return X, Y, T, dx, dy


def extraire_colonne(T, i):
    Nx, Ny = T.shape
    colonne = np.zeros(Ny)
    for j in range(1, Ny+1):
        colonne[j-1] = T[Nx - i, j-1]
    return colonne



def extraire_ligne(T, j):
    Nx, Ny = T.shape
    ligne = np.zeros(Nx)
    for i in range(1, Nx+1):
        ligne[i-1] = T[Nx - i, j-1]
    return ligne



if __name__ == "__main__":
    X, Y, T, dx, dy = maillage_temperature(5, 5, 10, 10, 0)

    T = np.arange(1, 17).reshape((4,4))

    print(extraire_colonne(T, 1))  # colonne physique 1
    print(extraire_ligne(T, 1))    # ligne physique 1



