import numpy as np
import calculs


def test_calcul_omega_bords():

    # --- Paramètres du maillage
    Nx, Ny = 5, 5
    Lx, Ly = 1.0, 1.0
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    # --- Maillage
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    psi = np.zeros((Nx, Ny))
    omega = np.zeros((Nx, Ny))

    # --- Champ psi analytique : psi = x^2 + y^2
    for i in range(Nx):
        for j in range(Ny):
            psi[i, j] = x[i]**2 + y[j]**2

    # --- Calcul de omega aux bords
    omega = calculs.calcul_omega_bords(psi, omega, dx, dy)

    # --- Affichage pour inspection
    print("Champ psi :\n", psi)
    print("\nOmega après calcul aux bords :\n", omega)

    # --- Vérifications simples
    # Intérieur doit rester nul
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            assert omega[i, j] == 0.0, "Erreur : omega intérieur modifié"

    print("\n✅ Test réussi : omega calculé uniquement aux frontières")
