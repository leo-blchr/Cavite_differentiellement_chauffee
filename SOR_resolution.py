def resolution_SOR(psi,omega,gamma0,dx,dy,Nx,Ny):
    (N_x, N_y) = np.shape(phi)
    beta=dx/dy
    
    indices = []

    for k in range(2, (Nx-2) + (Ny-2) + 1):  # diagonales pour i,j ≥ 1
        for i in range(1, Nx-1):
            j = k - i
            if 1 <= j <= Ny-2:
                indices.append((i,j))
                
    for (i,j) in indices:
        psi [i,j]= 1 / (2*(1+beta*beta)) * (psi [i+1,j])+psi [i-1,j] + beta*beta * (psi [i,j+1] +psi [i,j-1]) -dx *dx *omega [i, j]
    
    return(psi)



    
   
    
    