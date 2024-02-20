import numpy as np
def We_function(w, r, smatrix, alpha, gamma, mtauk, mtaun, mtauo, rho, g, ce, cf, ns, ntau):
    #demand for capital
    k_bar = (alpha / (r * (1 + mtauk))) ** ((1 - gamma) / (1 - gamma - alpha)) * \
        (gamma / (w * (1 + mtaun))) ** (gamma / (1 - gamma - alpha)) * \
        (smatrix * (1 - mtauo)) ** (1 / (1 - alpha - gamma))
    # Demand of labor
    n_bar = (1 + mtauk) * r * gamma / ((1 + mtaun) * w * alpha) * k_bar

    # Substitute kbar, nbar in pi and compute W(s,kbar(s,theta))
    pi_bar = (1 - mtauo) * smatrix * k_bar ** alpha * n_bar ** gamma - (1 + mtaun) * w * n_bar - (1 + mtauk) * r * k_bar - cf

    # Calculate W(s,tau)
    # Initial guess of W(s,tau) value function
    W = pi_bar / (1 - rho)
    xbar = np.zeros((ns, ntau))
    for i in range(ns):
        for j in range(ntau):
            if W[i, j] >= 0:
                xbar[i, j] = 1

    # Compute expected value of making a draw (s,theta) from G(s,theta).
    We = np.sum(W * g * xbar) - ce
    return We,k_bar,n_bar,pi_bar,W,xbar