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