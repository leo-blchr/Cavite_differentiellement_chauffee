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

    return T, dx, dy


def maillage_psi_omega(Lx, Ly, Nx, Ny, psi_init, omega_init):
    """
    Crée un maillage 2D identique pour la fonction de courant psi
    et la vorticité omega sur [0,Lx]x[0,Ly]

    Returns:
        psi, omega : champs (Nx, Ny)
        dx, dy     : pas du maillage
    """

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    psi = psi_init * np.ones((Nx, Ny))
    omega = omega_init * np.ones((Nx, Ny))

    return psi, omega, dx, dy



if __name__ == "__main__":
    T, dx, dy = maillage_temperature(5, 5, 10, 10, 0)



