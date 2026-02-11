import calculs
import numpy as np
dy=0.01
################## Définition des tests ######################

a=[1,1,2,3,4,5]
b=[3,4,5,0,1,2]
c=[2,2,2,2,2,2]
d=[3,2,1,6,5,4]

def test_Thomas():
    f=calculs.algorithme_thomas(a,b,c,d)
    f=np.array(f)
    assert( max(max(f-np.array([-72,-25,117/4,43/2,122/9,56/9])),-min(f-np.array([-72,-25,117/4,43/2,122/9,56/9])))<=0.001)


Psi=np.array([[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]])
u=1/dy*np.array([[0,-1/2,-1,0],[0,-3/2,-2,0],[0,1/2,1,0],[0,3/2,2,0]])


def test_thomas():
    # Matrice tridiagonale
    b = [4, 5, 6, 7]      # diagonale principale
    a = [1, 2, 3, 4]      # sous-diagonale (n-1 = 3)
    c = [2, 3, 4, 5]      # sur-diagonale (n-1 = 3)
    d = [5, 6, 7, 8]      # second membre

    n = len(b)
    A = np.zeros((n, n))

    np.fill_diagonal(A, b)
    np.fill_diagonal(A[1:], a[:-1])
    np.fill_diagonal(A[:, 1:], c[:-1])

    # Résolution
    x_thomas = calculs.algorithme_thomas(A.copy(), np.array(d, dtype=float))
    x_numpy = np.linalg.solve(A, d)

    print("Matrice A :\n", A)
    print("Vecteur d :", d)
    print("Solution Thomas :", x_thomas)
    print("Solution NumPy  :", x_numpy)
    print("Erreur max :", np.max(np.abs(x_thomas - x_numpy)))


def test_Speed_Computation():
    (un,vn)=calculs.Speed_Computation(Psi)
    assert(max(max(un-u),-min(un-u)))


def test_SOR():
    # La fonction fonctionne correctement selon ce qu'on attend pour un cas 4*4
    
    Nx=4
    Ny=4
    psin=np.array([[0.,0.,0.,0.],[0.,1.,1.,0.],[0.,1.,1.,0.],[0.,0.,0.,0.]])
    omegan=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    dx=1
    dy=1
    Psi=calculs.resolution_SOR_2(psin,omegan,gamma0=1.725, dx=dx, dy=dy)
    assert np.isclose(Psi[1,1], -1/8, atol=1e-8)
    assert np.isclose(Psi[1,2], -35/64, atol=1e-8)
    assert np.isclose(Psi[2,1], -35/64, atol=1e-8)


def test_SOR_2():
    Nx = 4
    Ny = 4
    dx = 1
    dy = 1
    gamma = 1.725

    psi = np.array([[0.,0.,0.,0.],
                    [0.,1.,1.,0.],
                    [0.,1.,1.,0.],
                    [0.,0.,0.,0.]])

    omega = np.ones((Nx, Ny))

    Psi = calculs.resolution_SOR_2(psi.copy(), omega, gamma, dx, dy)
    print(Psi)

    assert np.isclose(Psi[1,1], -0.29375, atol=1e-12)
    assert np.isclose(Psi[1,2], -0.8516796875, atol=1e-12)
    assert np.isclose(Psi[2,1], -0.8516796875, atol=1e-12)
    assert np.isclose(Psi[2,2], -1.89082373046875, atol=1e-12)




 
    















######### Execution ###############

test_SOR_2()