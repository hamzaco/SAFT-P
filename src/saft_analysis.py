#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def find_zero_crossings(x, d2A, tol=1e-8):

    # 1) copy and threshold small values to zero
    d2 = np.asarray(d2A, dtype=float).copy()
    d2[np.abs(d2) < tol] = 0.0

    # 2) compute sign array and fill any zeros by neighbor signs
    signs = np.sign(d2)
    zero_locs = np.where(signs == 0)[0]
    for z in zero_locs:
        if z + 1 < len(signs):
            signs[z] = signs[z+1]
        else:
            signs[z] = signs[z-1]

    # 3) find raw sign‐change indices
    cross_idxs = np.where(signs[:-1] * signs[1:] < 0)[0]

    # 4) linearly interpolate the crossing points
    x_crossings = []
    for i in cross_idxs:
        x0, x1 = x[i], x[i+1]
        y0, y1 = d2A[i],  d2A[i+1]
        if y1 != y0:
            xc = x0 - y0 * (x1 - x0) / (y1 - y0)
        else:
            xc = 0.5*(x0 + x1)
        x_crossings.append(xc)

    return cross_idxs, x_crossings

def residual_free_energy_multi_numba(phi, β, z,g,
                                     patch_to_species, ε_patch, m_patch,eps_ns=0):
    # 1) Ideal mixing
    A_mix = 0.0
    for i in range(phi.shape[0]):
        A_mix += phi[i]*np.log(phi[i]/g[i]) # + (1-phi[i])*np.log(1-phi[i])
    # 2) Association
    X = solve_association_multi_numba(phi, patch_to_species,ε_patch, m_patch, β, z)
    A_assoc = 0.0
    P = m_patch.shape[0]
    for p in range(P):
        s = patch_to_species[p]
        A_assoc += m_patch[p]* phi[s]  * (np.log(X[p]) - 0.5*X[p] + 0.5)
    return (A_mix+ A_assoc ),A_mix

def solve_association_multi_numba(phi, patch_to_species, eps_patch, m_patch, beta, z,
                                  tol=1e-8, max_iter=5000, alpha=0.2):
    P = eps_patch.shape[0]
    Delta = (np.exp(beta * eps_patch) - 1.0) / z  # keep your 1/4 if that is your lattice normalization
    phi_p = phi[patch_to_species]

    X = np.ones(P)
    X_new = np.empty(P)
    denom = np.empty(P)

    for _ in range(max_iter):
        # build denom
        for j in range(P):
            tmp = 0.0
            for k in range(P):
                tmp += Delta[j, k] * phi_p[k] * X[k] * m_patch[k]
            val = 1.0 + tmp                 # <-- use z
            if val <= 1e-16:                    # if this triggers often, your model/inputs are problematic
                val = 1e-16
            denom[j] = val

        # fixed-point map
        for j in range(P):
            Xfp = 1.0 / denom[j]
            Xn  = alpha * Xfp + (1.0 - alpha) * X[j]
            # clip (fine), but don’t let this mask persistent denom pathologies
            if Xn < 1e-16: Xn = 1e-16
            if Xn > 1.0:   Xn = 1.0
            X_new[j] = Xn

        maxdiff = np.max(np.abs(X_new - X))
        if not np.isfinite(maxdiff):
            return X  # or raise

        if maxdiff < tol:
            return X_new.copy()

        X[:] = X_new

    return X

