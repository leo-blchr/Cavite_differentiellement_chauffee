import numpy as np


"""


def spatial_derivative_2D(f_flat, x, y):
    
    # Calcule les dérivées spatiales ∂f/∂x et ∂f/∂y en 2D

    # Parameters
    # ----------
    # f_flat = [f(1,1), ..., f(1,Nx), f(2,1), ..., f(Ny,Nx)]
    # x : 1D array (Nx,)
    #     Coordonnées en x
    # y : 1D array (Ny,)
    #     Coordonnées en y

    # Returns
    # -------
    # dfdx_flat : 1D array
    #     ∂f/∂x aplati
    # dfdy_flat : 1D array
    #     ∂f/∂y aplati
    
    Nx = len(x)
    Ny = len(y)

    # reshape → f[y, x]
    f = np.asarray(f_flat).reshape((Ny, Nx))

    dfdx = np.zeros_like(f)
    dfdy = np.zeros_like(f)

    # ∂f/∂x (x varie le plus vite → axe 1)
    for j in range(Ny):
        for i in range(1, Nx-1):
            dfdx[j, i] = (f[j, i+1] - f[j, i-1]) / (x[i+1] - x[i-1])

        # bords
        dfdx[j, 0]    = (f[j, 1] - f[j, 0]) / (x[1] - x[0])
        dfdx[j, -1]   = (f[j, -1] - f[j, -2]) / (x[-1] - x[-2])

    # ∂f/∂y (y varie lentement → axe 0)
    for i in range(Nx):
        for j in range(1, Ny-1):
            dfdy[j, i] = (f[j+1, i] - f[j-1, i]) / (y[j+1] - y[j-1])

        # bords
        dfdy[0, i]    = (f[1, i] - f[0, i]) / (y[1] - y[0])
        dfdy[-1, i]   = (f[-1, i] - f[-2, i]) / (y[-1] - y[-2])

    return dfdx.ravel(), dfdy.ravel()

"""


def dfdx_at_x(f_flat, x, y, ix):
    Nx = len(x)
    Ny = len(y)

    f = np.asarray(f_flat).reshape((Nx, Ny))
    dfdx = np.zeros(Ny)

    if ix == 0:
        dx = x[1] - x[0]
        dfdx[:] = (f[1, :] - f[0, :]) / dx

    elif ix == Nx - 1:
        dx = x[-1] - x[-2]
        dfdx[:] = (f[-1, :] - f[-2, :]) / dx

    else:
        dx = x[ix+1] - x[ix-1]
        dfdx[:] = (f[ix+1, :] - f[ix-1, :]) / dx

    return dfdx




def dfdy_at_y(f_flat, x, y, iy):
    Nx = len(x)
    Ny = len(y)

    f = np.asarray(f_flat).reshape((Nx, Ny))
    dfdy = np.zeros(Nx)

    if iy == 0:
        dy = y[1] - y[0]
        dfdy[:] = (f[:, 1] - f[:, 0]) / dy

    elif iy == Ny - 1:
        dy = y[-1] - y[-2]
        dfdy[:] = (f[:, -1] - f[:, -2]) / dy

    else:
        dy = y[iy+1] - y[iy-1]
        dfdy[:] = (f[:, iy+1] - f[:, iy-1]) / dy

    return dfdy



def algorithme_thomas(A, d):
    """
    Résout Ax = d pour une matrice tridiagonale A
    """
    import numpy as np

    A = A.astype(float)
    d = d.astype(float)

    n = A.shape[0]

    # extraction des diagonales
    a = np.zeros(n-1)  # sous-diagonale
    b = np.zeros(n)    # diagonale principale
    c = np.zeros(n-1)  # sur-diagonale

    for i in range(n):
        b[i] = A[i, i]
        if i > 0:
            a[i-1] = A[i, i-1]
        if i < n-1:
            c[i] = A[i, i+1]

    # Thomas classique
    for i in range(1, n):
        m = a[i-1] / b[i-1]
        b[i] -= m * c[i-1]
        d[i] -= m * d[i-1]

    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]

    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]

    return x


if __name__ == "__main__":
    d = np.array([1, 0, 0, 1])

    A = np.array([[2, -1, 0, 0],
              [-1, 2, -1, 0],
              [0, -1, 2, -1],
              [0, 0, -1, 2]])
    x_expected = np.linalg.solve(A, d)

    x = algorithme_thomas(A,d)
    print(x, x_expected)
