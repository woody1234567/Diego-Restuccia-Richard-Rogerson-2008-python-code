import numpy as np
from We_function import We_function
def Sub_We_function(taus,alpha, gamma, g, Kbe, lambda_exit, mtau, mtauo, mtauk, mtaun, ns, ntau, S, s_grid, smatrix, tau, rho, ce, cf, r):
    tauv = [-taus, 0, tau]

    # The state space of plants
    # is described by (s,tau). Construct auxiliary matrices
    # for s and tau grid.
    smatrix = S * np.outer( s_grid , np.ones((1, ntau)))
    mtau = np.ones((ns, 1)) * tauv

    # type2 index describes the type of subsidy policy the government
    # implements: 1 - output, 2 - capital, 3 - labor.
    type2 = 1

    if type2 == 1:
        mtauo = mtau
        mtauk = np.zeros((ns, ntau))
        mtaun = np.zeros((ns, ntau))
    elif type2 == 2:
        mtauo = np.zeros((ns, ntau))
        mtauk = mtau
        mtaun = np.zeros((ns, ntau))
    else:
        mtauo = np.zeros((ns, ntau))
        mtauk = np.zeros((ns, ntau))
        mtaun = mtau

    # Compute equilibrium wage that makes net expected profits equal
    # to 0 for potential entering plants We=0.

    # Bisection on equilibrium wage

    # Initial guess
    w0 = 1
    Output_temp_0 = We_function(w0, r, smatrix, alpha, gamma, mtauk, mtaun, mtauo, rho, g, ce, cf, ns, ntau)
    We0=Output_temp_0[0]
    w1 = 1.5
    Output_temp_1 = We_function(w1, r, smatrix, alpha, gamma, mtauk, mtaun, mtauo, rho, g, ce, cf, ns, ntau)
    We1=Output_temp_1[0]

    while We0 * We1 > 0:
        if We0 < 0:
            w0 *= 0.5
            Output_temp_0 = We_function(w0, r, smatrix, alpha, gamma, mtauk, mtaun, mtauo, rho, g, ce, cf, ns, ntau)
            We0=Output_temp_0[0]
        if We1 > 0:
            w1 *= 1.5
            Output_temp_1 = We_function(w1, r, smatrix, alpha, gamma, mtauk, mtaun, mtauo, rho, g, ce, cf, ns, ntau)
            We1=Output_temp_1[0]
    iconv2 = 0
    tol2 = 0.0000001
    maxit2 = 100
    it2 = 1

    while iconv2 == 0 and it2 <= maxit2:
        w = (w0 + w1) / 2
        Output_temp = We_function(w, r, smatrix, alpha, gamma, mtauk, mtaun, mtauo, rho, g, ce, cf, ns, ntau)
        We=Output_temp[0]
        if abs(We) < tol2:
            iconv2 = 1
            # print('Zero profit condition satisfied in')
            # print(it2)
        else:
            if We * We1 > 0:
                w1 = w
            else:
                w0 = w
            it2 += 1

    if it2 >= maxit2:
        print('Warning: Zero profit condition not satisfied')
    
    # decode the output
    k_bar=Output_temp[1] # optimal capital
    n_bar=Output_temp[2] 
    xbar=Output_temp[5]    

    # Compute the normalized invariant distribution of plants.
    muhat = 1. / lambda_exit * xbar * g

    # Find mass of entry that clears the labor market.
    N = np.sum(np.sum(n_bar * muhat))
    E = 1 / N
    mu = muhat * E

    # Compute aggregate capital and capital-to-output ratio
    K = np.sum(np.sum(k_bar * mu))
    Y = np.sum(np.sum(smatrix * k_bar ** alpha * n_bar ** gamma * mu))

    # Compute resid as difference between the actual and target revenue to
    # output ratio.
    resid = K / Kbe - 1
    return resid, k_bar, n_bar, xbar, E, N, mu, w, Y

