# simulate_functions_min.py
# Minimal dependencies to run simulate_lattice_numba (Numba implementation)
import numpy as np
import numba as nb
class Particle:
    def __init__(self,index,patches=[],mu=0):
        self.patches=patches
        self.mu=mu
        self.index=index
        
    def get(self,i): 
        return self.patches[i]
    def set_mu(self,mu):
        self.mu=mu

def get_species_list_ind(species):
    letter_to_index = {}  # mapping from letter to unique integer index
    next_index = 1
    particles = []
    letter_to_index['0']=0
    ind=0
    for s in species:
        n = len(s)
        # Create a base patch array of length 4 filled with zeros (0 means "no $
        base_array = np.zeros(4, dtype=int)
        # Right-align the string: fill positions 4 - n to 3 with the correspond$
        for i, letter in enumerate(s):
            if letter not in letter_to_index:
                letter_to_index[letter] = next_index
                next_index += 1
            base_array[4 - n + i] = letter_to_index[letter]
       
        # Generate all four rotations of the base array.
        unique_rotations = set()
        rotation_list = []  # to preserve an order (if needed)
        for r in range(4):
            rotated = np.roll(base_array, -r)  # rotate left by r positions
            tup = tuple(rotated.tolist())
            if tup not in unique_rotations:
                unique_rotations.add(tup)
                rotation_list.append(rotated)
        
        # For each distinct rotation, create a Particle object.
        for rot in rotation_list:
            particles.append(Particle(ind,patches=np.array(rot)))
            ind+=1
    return particles

# -------------------- precompute tables --------------------
def precompute_bond_table_from_patches(J_patch, patches_by_species):
    """
    J_patch: (n_patch, n_patch) energy lookup for patch ids
    patches_by_species: (n_species, 4) patch ids in [N,E,S,W]
    Returns:
        Bdir[dir, si, sj] = bond energy between species si (center) and species sj (neighbor)
        dir = 0(N),1(E),2(S),3(W), paired with opposite faces.
    """
    Jp = np.asarray(J_patch, dtype=np.float64)
    P  = np.asarray(patches_by_species, dtype=np.int16)
    n_species = P.shape[0]
    B = np.empty((4, n_species, n_species), dtype=np.float64)
    for si in range(n_species):
        piN, piE, piS, piW = map(int, P[si])
        for sj in range(n_species):
            pjN, pjE, pjS, pjW = map(int, P[sj])
            B[0, si, sj] = Jp[piN, pjS]  # N meets S
            B[1, si, sj] = Jp[piE, pjW]  # E meets W
            B[2, si, sj] = Jp[piS, pjN]  # S meets N
            B[3, si, sj] = Jp[piW, pjE]  # W meets E
    return B

def make_mask(n_species, indices):
    mask = np.zeros(n_species, dtype=np.uint8)
    for idx in np.asarray(indices, dtype=np.int64).ravel():
        if 0 <= int(idx) < n_species:
            mask[int(idx)] = 1
    return mask

# -------------------- numba helpers --------------------
@nb.njit(cache=True, fastmath=True)
def _push_unique(buf, m, x):
    for i in range(m):
        if buf[i] == x:
            return m
    buf[m] = x
    return m + 1

@nb.njit(cache=True, fastmath=True)
def _local_energy_site(k, lat, nbr, Bdir):
    s = lat[k]
    n0 = lat[nbr[k, 0]]
    n1 = lat[nbr[k, 1]]
    n2 = lat[nbr[k, 2]]
    n3 = lat[nbr[k, 3]]
    return 0.5 * (Bdir[0, s, n0] + Bdir[1, s, n1] + Bdir[2, s, n2] + Bdir[3, s, n3])

@nb.njit(cache=True, fastmath=True)
def _bond_counts_right(s, sr, empty_index, P):
    if s == empty_index or sr == empty_index:
        return 0, 0
    tot = 1
    sp = 1 if (P[s, 1] == 1 and P[sr, 3] == 1) else 0
    return tot, sp

@nb.njit(cache=True, fastmath=True)
def _bond_counts_down(s, sd, empty_index, P):
    if s == empty_index or sd == empty_index:
        return 0, 0
    tot = 1
    sp = 1 if (P[s, 2] == 1 and P[sd, 0] == 1) else 0
    return tot, sp

# -------------------- core MC kernel --------------------
@nb.njit(cache=True, fastmath=True)
def _mc_lattice_single_site_numba(
    lat2d_int, Bdir, P, mu,
    mask_mu, mask_ns, mask_ew,
    empty_index,
    eps_ns0, eps_sp0,
    simulation_steps, beta,
    snapshot_interval, buffer_size,
    block_size
):
    H, W = lat2d_int.shape
    N = H * W
    n_species = mu.shape[0]
    invN = 1.0 / N

    # neighbors (periodic), flattened index k=i*W+j
    nbr = np.empty((N, 4), dtype=np.int32)  # 0=N,1=E,2=S,3=W
    for i in range(H):
        im = (i - 1) % H
        ip = (i + 1) % H
        for j in range(W):
            jm = (j - 1) % W
            jp = (j + 1) % W
            k = i * W + j
            nbr[k, 0] = im * W + j
            nbr[k, 1] = i  * W + jp
            nbr[k, 2] = ip * W + j
            nbr[k, 3] = i  * W + jm

    lat = lat2d_int.ravel().copy()

    # species counts
    counts = np.zeros(n_species, dtype=np.int64)
    for k in range(N):
        counts[lat[k]] += 1

    # Nocc, Nmu
    Nocc = N - counts[empty_index]
    Nmu = 0
    for s in range(n_species):
        if mask_mu[s] != 0:
            Nmu += counts[s]

    # Q bookkeeping: N_NS, N_EW (counts by species ids)
    N_NS = 0
    N_EW = 0
    for s in range(n_species):
        if mask_ns[s] != 0:
            N_NS += counts[s]
        if mask_ew[s] != 0:
            N_EW += counts[s]

    # initial bond counts (right+down only)
    C_tot = 0
    C_sp  = 0
    for i in range(H):
        for j in range(W):
            k = i * W + j
            s = lat[k]
            sr = lat[nbr[k, 1]]
            sd = lat[nbr[k, 2]]
            t, sp = _bond_counts_right(s, sr, empty_index, P)
            C_tot += t
            C_sp  += sp
            t, sp = _bond_counts_down(s, sd, empty_index, P)
            C_tot += t
            C_sp  += sp
    C_ns = C_tot - C_sp

    # energy field and total energy (bond + chemical potential term)
    e_sites = np.empty(N, dtype=np.float64)
    energy = 0.0
    for k in range(N):
        ev = _local_energy_site(k, lat, nbr, Bdir)
        e_sites[k] = ev
        energy += ev
    for k in range(N):
        energy += mu[lat[k]]

    # ring buffer sizing
    total_samples = 1 + simulation_steps // snapshot_interval
    n_keep = buffer_size
    if n_keep < 1:
        n_keep = 1
    if n_keep > total_samples:
        n_keep = total_samples

    energies_buf  = np.empty(n_keep, dtype=np.float64)
    densities_buf = np.empty((n_keep, n_species), dtype=np.float64)
    N_mu_buf      = np.empty(n_keep, dtype=np.int64)
    Nocc_buf      = np.empty(n_keep, dtype=np.int64)
    C_sp_buf      = np.empty(n_keep, dtype=np.int64)
    C_ns_buf      = np.empty(n_keep, dtype=np.int64)
    u_buf         = np.empty(n_keep, dtype=np.float64)
    Q_buf         = np.empty(n_keep, dtype=np.float64)

    # acceptance ratios per block
    n_blocks = simulation_steps // block_size
    acc_blocks = np.empty(n_blocks, dtype=np.float64)
    accepted_in_block = 0

    tries = 0
    accepts = 0

    # local update buffers (site + 4 nbrs)
    buf5 = np.empty(5, dtype=np.int32)
    oldE = np.empty(5, dtype=np.float64)
    newE = np.empty(5, dtype=np.float64)

    sample_count = 0

    def record():
        nonlocal sample_count
        i = sample_count % n_keep
        energies_buf[i] = energy
        for s in range(n_species):
            densities_buf[i, s] = counts[s] * invN
        N_mu_buf[i]  = Nmu
        Nocc_buf[i]  = Nocc
        C_sp_buf[i]  = C_sp
        C_ns_buf[i]  = C_ns
        u_buf[i]     = -((C_ns+C_sp) * eps_ns0 + C_sp * eps_sp0) * invN
        Q_buf[i]     = (N_NS - N_EW) * invN
        sample_count += 1

    record()  # t=0

    for step in range(simulation_steps):
        tries += 1

        k = np.random.randint(N)
        old_s = lat[k]
        new_s = np.random.randint(n_species)

        accepted = False

        if new_s == old_s:
            accepted = True
        else:
            # affected sites: k and its 4 neighbors
            m = 0
            m = _push_unique(buf5, m, k)
            for d in range(4):
                m = _push_unique(buf5, m, nbr[k, d])

            for t in range(m):
                kk = buf5[t]
                oldE[t] = e_sites[kk]

            # bonds counted once that can change:
            kr = nbr[k, 1]
            kd = nbr[k, 2]
            kw = nbr[k, 3]
            kn = nbr[k, 0]

            tot0, sp0 = _bond_counts_right(old_s, lat[kr], empty_index, P)
            tot1, sp1 = _bond_counts_down(old_s,  lat[kd], empty_index, P)
            tot2, sp2 = _bond_counts_right(lat[kw], old_s, empty_index, P)
            tot3, sp3 = _bond_counts_down(lat[kn], old_s, empty_index, P)
            Ctot_old = tot0 + tot1 + tot2 + tot3
            Csp_old  = sp0  + sp1  + sp2  + sp3

            # propose
            lat[k] = new_s

            tot0, sp0 = _bond_counts_right(new_s, lat[kr], empty_index, P)
            tot1, sp1 = _bond_counts_down(new_s,  lat[kd], empty_index, P)
            tot2, sp2 = _bond_counts_right(lat[kw], new_s, empty_index, P)
            tot3, sp3 = _bond_counts_down(lat[kn], new_s, empty_index, P)
            Ctot_new = tot0 + tot1 + tot2 + tot3
            Csp_new  = sp0  + sp1  + sp2  + sp3

            # local energy delta
            dE_bonds = 0.0
            for t in range(m):
                kk = buf5[t]
                v = _local_energy_site(kk, lat, nbr, Bdir)
                newE[t] = v
                dE_bonds += (v - oldE[t])

            dmu = mu[new_s] - mu[old_s]
            dPhi = dE_bonds + dmu

            if (dPhi <= 0.0) or (np.random.rand() < np.exp(-beta * dPhi)):
                accepted = True

                # commit energy field + total energy
                for t in range(m):
                    e_sites[buf5[t]] = newE[t]
                energy += dE_bonds + dmu

                # commit counts
                counts[old_s] -= 1
                counts[new_s] += 1

                # occupancy
                if old_s == empty_index and new_s != empty_index:
                    Nocc += 1
                elif old_s != empty_index and new_s == empty_index:
                    Nocc -= 1

                # mu-coupled count
                Nmu += (1 if mask_mu[new_s] != 0 else 0) - (1 if mask_mu[old_s] != 0 else 0)

                # Q bookkeeping
                N_NS += (1 if mask_ns[new_s] != 0 else 0) - (1 if mask_ns[old_s] != 0 else 0)
                N_EW += (1 if mask_ew[new_s] != 0 else 0) - (1 if mask_ew[old_s] != 0 else 0)

                # contact counts
                C_tot += (Ctot_new - Ctot_old)
                C_sp  += (Csp_new  - Csp_old)
                C_ns = C_tot - C_sp
            else:
                # reject: revert state and restore local e_sites
                lat[k] = old_s
                for t in range(m):
                    e_sites[buf5[t]] = oldE[t]

        if accepted:
            accepts += 1
            accepted_in_block += 1

        if (step + 1) % block_size == 0 and (step // block_size) < n_blocks:
            acc_blocks[step // block_size] = accepted_in_block / block_size
            accepted_in_block = 0

        if (step + 1) % snapshot_interval == 0:
            record()

    return (energies_buf, densities_buf,
            N_mu_buf, Nocc_buf, C_sp_buf, C_ns_buf, u_buf, Q_buf,
            sample_count, n_keep, acc_blocks,
            tries, accepts,
            lat.reshape(H, W))

# -------------------- python rollout for ring buffers --------------------
def _rollout_ring_1d(buf, count):
    n = len(buf); k = min(count, n)
    if count <= n:
        return buf[:k].copy()
    start = count % n
    return np.concatenate((buf[start:], buf[:start]), axis=0)

def _rollout_ring_2d(buf, count):
    n = buf.shape[0]; k = min(count, n)
    if count <= n:
        return buf[:k].copy()
    start = count % n
    return np.concatenate((buf[start:], buf[:start]), axis=0)

# -------------------- public entrypoint --------------------
def simulate_lattice_numba(
    lattice_int, J_patch, patches_by_species, mu_table,
    empty_index,
    species_mu_indices,
    eps_ns0, eps_sp0,
    simulation_steps=1_000_000, beta=1.0,
    snapshot_interval=1000,
    buffer_size=500_000,
    block_size=1000,
    ns_species_ids=(0,),
    ew_species_ids=(1,),
):
    """
    Monte Carlo single-site identity updates on a 2D periodic lattice.

    Inputs
      lattice_int        : (Ly,Lx) int species ids
      J_patch            : (n_patch,n_patch) energy lookup by patch ids
      patches_by_species : (n_species,4) patch ids per species in [N,E,S,W]
      mu_table           : (n_species,) chemical potentials coupled to identity
      empty_index        : species id for vacancies
      species_mu_indices : iterable of species ids counted in N_mu (for reweighting)
      eps_ns0, eps_sp0   : scalars used ONLY to report u = -(C_ns*eps_ns0 + C_sp*eps_sp0)/N_sites
      ns_species_ids, ew_species_ids : define Q = (N_NS - N_EW)/N_sites for diagnostics

    Outputs
      snap_counts, energies, densities, lat_final
    """
    lattice_int = np.asarray(lattice_int, dtype=np.int64)
    P  = np.asarray(patches_by_species, dtype=np.int16)
    mu = np.asarray(mu_table, dtype=np.float64)

    # safety
    if lattice_int.min() < 0 or lattice_int.max() >= P.shape[0]:
        raise ValueError("lattice_int has species ids outside patches_by_species")
    if mu.shape[0] != P.shape[0]:
        raise ValueError("mu_table must have length n_species == patches_by_species.shape[0]")
    if not (0 <= int(empty_index) < P.shape[0]):
        raise ValueError("empty_index out of range")
    if int(P.max(initial=0)) >= np.asarray(J_patch).shape[0]:
        raise ValueError("J_patch too small for patch ids in patches_by_species")

    Bdir = precompute_bond_table_from_patches(J_patch, P)

    n_species = P.shape[0]
    mask_mu = make_mask(n_species, species_mu_indices)
    mask_ns = make_mask(n_species, ns_species_ids)
    mask_ew = make_mask(n_species, ew_species_ids)

    (energies_buf, densities_buf,
     N_mu_buf, Nocc_buf, C_sp_buf, C_ns_buf, u_buf, Q_buf,
     sample_count, n_keep, acc_blocks,
     tries, accepts,
     lat_final) = _mc_lattice_single_site_numba(
        lattice_int, Bdir, P, mu,
        mask_mu, mask_ns, mask_ew,
        int(empty_index),
        float(eps_ns0), float(eps_sp0),
        int(simulation_steps), float(beta),
        int(snapshot_interval), int(buffer_size),
        int(block_size)
    )

    energies  = _rollout_ring_1d(energies_buf,  sample_count)
    densities = _rollout_ring_2d(densities_buf, sample_count)

    N_sites = int(lattice_int.size)
    N_mu_arr  = _rollout_ring_1d(N_mu_buf,  sample_count).astype(float)
    Nocc_arr  = _rollout_ring_1d(Nocc_buf,  sample_count).astype(float)
    C_sp_arr  = _rollout_ring_1d(C_sp_buf,  sample_count).astype(float)
    C_ns_arr  = _rollout_ring_1d(C_ns_buf,  sample_count).astype(float)
    u_arr     = _rollout_ring_1d(u_buf,     sample_count).astype(float)
    Q_arr     = _rollout_ring_1d(Q_buf,     sample_count).astype(float)

    snap_counts = dict(
        N_sites=N_sites,
        Lx=int(lattice_int.shape[1]),
        Ly=int(lattice_int.shape[0]),
        N_mu=N_mu_arr,
        rho_mu=N_mu_arr / float(N_sites),
        Nocc_total=Nocc_arr,
        rho_total=Nocc_arr / float(N_sites),
        C_ns=C_ns_arr,
        C_sp=C_sp_arr,
        u=u_arr,
        Q=Q_arr,
        acceptance_ratio_blocks=np.array(acc_blocks, dtype=float),
        tries=int(tries),
        accepts=int(accepts),
        accept_rate=float(accepts / tries) if tries else 0.0,
        snapshot_interval_steps=int(snapshot_interval),
        block_size=int(block_size),
    )

    return snap_counts, energies, densities, lat_final
