def resolution_SOR(psi, omega, gamma0, dx, dy):
    Nx, Ny = psi.shape
    beta = dx/dy
    
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            psi_new = (psi[i+1,j] + psi[i-1,j] + beta**2*(psi[i,j+1]+psi[i,j-1]) - dx**2*omega[i,j]) / (2*(1+beta**2))
            # SOR relaxation
            psi[i,j] = (1-gamma0)*psi[i,j] + gamma0*psi_new

    return psi