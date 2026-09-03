# simulate_functions_triangular.py
# MC simulation on a 2D triangular lattice with 6-patch particles (Numba)
import numpy as np
import numba as nb


class Particle:
    def __init__(self, index, patches=[], mu=0):
        self.patches = patches
        self.mu = mu
        self.index = index

    def get(self, i):
        return self.patches[i]

    def set_mu(self, mu):
        self.mu = mu


# ------------------------------------------------------------------ #
#  Patch ordering convention (triangular / hexagonal neighbours):
#    index 0 = E,  1 = NE,  2 = NW,  3 = W,  4 = SW,  5 = SE
#  Opposite direction:  opp(d) = (d + 3) % 6
# ------------------------------------------------------------------ #

def get_species_list_ind_hex(species):
    """Build species list for 6-patch (triangular lattice) particles.

    Each string in *species* encodes the patch letters; short strings are
    right-aligned into a length-6 array (leading positions filled with '0'
    = no patch).  All distinct 60-degree rotations are generated.
    """
    letter_to_index = {}
    next_index = 1
    particles = []
    letter_to_index['0'] = 0
    ind = 0
    for s in species:
        n = len(s)
        base_array = np.zeros(6, dtype=int)
        for i, letter in enumerate(s):
            if letter not in letter_to_index:
                letter_to_index[letter] = next_index
                next_index += 1
            base_array[6 - n + i] = letter_to_index[letter]

        unique_rotations = set()
        rotation_list = []
        for r in range(6):
            rotated = np.roll(base_array, -r)
            tup = tuple(rotated.tolist())
            if tup not in unique_rotations:
                unique_rotations.add(tup)
                rotation_list.append(rotated)

        for rot in rotation_list:
            particles.append(Particle(ind, patches=np.array(rot)))
            ind += 1
    return particles


# -------------------- precompute tables ----------------------------- #

def precompute_bond_table_hex(J_patch, patches_by_species):
    """Bond table for the triangular lattice (6 directions).

    Returns Bdir[d, si, sj] = bond energy when species *si* sits at
    the centre and species *sj* is its neighbour in direction *d*.
    The facing patch of the neighbour is at direction (d+3)%6.
    """
    Jp = np.asarray(J_patch, dtype=np.float64)
    P  = np.asarray(patches_by_species, dtype=np.int16)
    n_species = P.shape[0]
    B = np.empty((6, n_species, n_species), dtype=np.float64)
    for si in range(n_species):
        for sj in range(n_species):
            for d in range(6):
                opp = (d + 3) % 6
                B[d, si, sj] = Jp[int(P[si, d]), int(P[sj, opp])]
    return B


def make_mask(n_species, indices):
    mask = np.zeros(n_species, dtype=np.uint8)
    for idx in np.asarray(indices, dtype=np.int64).ravel():
        if 0 <= int(idx) < n_species:
            mask[int(idx)] = 1
    return mask


# -------------------- numba helpers --------------------------------- #

@nb.njit(cache=True, fastmath=True)
def _push_unique(buf, m, x):
    for i in range(m):
        if buf[i] == x:
            return m
    buf[m] = x
    return m + 1


@nb.njit(cache=True, fastmath=True)
def _local_energy_site_hex(k, lat, nbr, Bdir):
    """Half of the pairwise bond energy contributed by site *k*."""
    s = lat[k]
    e = 0.0
    for d in range(6):
        e += Bdir[d, s, lat[nbr[k, d]]]
    return 0.5 * e


@nb.njit(cache=True, fastmath=True)
def _bond_counts_dir(s, sn, empty_index, P, d):
    """Bond in forward direction *d* from species *s* to neighbour *sn*.

    Returns (tot, sp):
        tot = 1 if both sites occupied, else 0
        sp  = 1 if both facing patches are type-1, else 0
    """
    if s == empty_index or sn == empty_index:
        return 0, 0
    tot = 1
    opp = (d + 3) % 6
    sp = 1 if (P[s, d] == 1 and P[sn, opp] == 1) else 0
    return tot, sp


# -------------------- core MC kernel -------------------------------- #

@nb.njit(cache=True, fastmath=True)
def _mc_triangular_single_site(
    lat2d_int, Bdir, P, mu,
    mask_mu, mask_ns, mask_ew,
    empty_index,
    eps_ns0, eps_sp0,
    simulation_steps, beta,
    snapshot_interval, buffer_size,
    block_size,
):
    H, W = lat2d_int.shape
    N = H * W
    n_species = mu.shape[0]
    invN = 1.0 / N

    # ---- triangular-lattice neighbour table (periodic) ----
    # Even row  (i%2==0): NE/NW/SW/SE at same or col-1
    # Odd  row  (i%2==1): NE/NW/SW/SE at col+1 or same
    nbr = np.empty((N, 6), dtype=np.int32)
    for i in range(H):
        im = (i - 1) % H
        ip = (i + 1) % H
        for j in range(W):
            jm = (j - 1 + W) % W
            jp = (j + 1) % W
            k = i * W + j
            if i % 2 == 0:
                nbr[k, 0] = i  * W + jp   # E
                nbr[k, 1] = im * W + j    # NE
                nbr[k, 2] = im * W + jm   # NW
                nbr[k, 3] = i  * W + jm   # W
                nbr[k, 4] = ip * W + jm   # SW
                nbr[k, 5] = ip * W + j    # SE
            else:
                nbr[k, 0] = i  * W + jp   # E
                nbr[k, 1] = im * W + jp   # NE
                nbr[k, 2] = im * W + j    # NW
                nbr[k, 3] = i  * W + jm   # W
                nbr[k, 4] = ip * W + j    # SW
                nbr[k, 5] = ip * W + jp   # SE

    lat = lat2d_int.ravel().copy()

    counts = np.zeros(n_species, dtype=np.int64)
    for k in range(N):
        counts[lat[k]] += 1

    Nocc = N - counts[empty_index]
    Nmu = 0
    for s in range(n_species):
        if mask_mu[s] != 0:
            Nmu += counts[s]

    N_NS = 0
    N_EW = 0
    for s in range(n_species):
        if mask_ns[s] != 0:
            N_NS += counts[s]
        if mask_ew[s] != 0:
            N_EW += counts[s]

    # Forward directions for single-counted bonds: E(0), SW(4), SE(5)
    # Their opposites W(3), NE(1), NW(2) are NOT included ⇒ each bond
    # is counted exactly once.
    FWD0 = 0
    FWD1 = 4
    FWD2 = 5

    # initial bond counts (forward directions only)
    C_tot = 0
    C_sp  = 0
    for i in range(H):
        for j in range(W):
            k = i * W + j
            s = lat[k]
            for fd in (FWD0, FWD1, FWD2):
                t, sp = _bond_counts_dir(s, lat[nbr[k, fd]], empty_index, P, fd)
                C_tot += t
                C_sp  += sp
    C_ns = C_tot - C_sp

    # energy field and total energy
    e_sites = np.empty(N, dtype=np.float64)
    energy = 0.0
    for k in range(N):
        ev = _local_energy_site_hex(k, lat, nbr, Bdir)
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

    n_blocks = simulation_steps // block_size
    acc_blocks = np.empty(n_blocks, dtype=np.float64)
    accepted_in_block = 0

    tries   = 0
    accepts = 0

    buf7 = np.empty(7, dtype=np.int32)   # site + up to 6 neighbours
    oldE = np.empty(7, dtype=np.float64)
    newE = np.empty(7, dtype=np.float64)

    sample_count = 0

    def record():
        nonlocal sample_count
        idx = sample_count % n_keep
        energies_buf[idx] = energy
        for s in range(n_species):
            densities_buf[idx, s] = counts[s] * invN
        N_mu_buf[idx]  = Nmu
        Nocc_buf[idx]  = Nocc
        C_sp_buf[idx]  = C_sp
        C_ns_buf[idx]  = C_ns
        u_buf[idx] = -((C_ns + C_sp) * eps_ns0 + C_sp * eps_sp0) * invN
        Q_buf[idx]     = (N_NS - N_EW) * invN
        sample_count  += 1

    record()  # t = 0

    for step in range(simulation_steps):
        tries += 1

        k     = np.random.randint(N)
        old_s = lat[k]
        new_s = np.random.randint(n_species)

        accepted = False

        if new_s == old_s:
            accepted = True
        else:
            # affected sites: k and its 6 neighbours
            m = 0
            m = _push_unique(buf7, m, k)
            for d in range(6):
                m = _push_unique(buf7, m, nbr[k, d])

            for t in range(m):
                oldE[t] = e_sites[buf7[t]]

            # --- old bond counts (6 bonds incident on k) ---
            # forward FROM k  (directions 0, 4, 5)
            Ctot_old = 0; Csp_old = 0
            for fd in (FWD0, FWD1, FWD2):
                t0, s0 = _bond_counts_dir(old_s, lat[nbr[k, fd]],
                                          empty_index, P, fd)
                Ctot_old += t0; Csp_old += s0
            # forward TO k  (W→E=0, NE→SW=4, NW→SE=5)
            t0, s0 = _bond_counts_dir(lat[nbr[k, 3]], old_s,
                                      empty_index, P, FWD0)
            Ctot_old += t0; Csp_old += s0
            t0, s0 = _bond_counts_dir(lat[nbr[k, 1]], old_s,
                                      empty_index, P, FWD1)
            Ctot_old += t0; Csp_old += s0
            t0, s0 = _bond_counts_dir(lat[nbr[k, 2]], old_s,
                                      empty_index, P, FWD2)
            Ctot_old += t0; Csp_old += s0

            # propose
            lat[k] = new_s

            # --- new bond counts ---
            Ctot_new = 0; Csp_new = 0
            for fd in (FWD0, FWD1, FWD2):
                t0, s0 = _bond_counts_dir(new_s, lat[nbr[k, fd]],
                                          empty_index, P, fd)
                Ctot_new += t0; Csp_new += s0
            t0, s0 = _bond_counts_dir(lat[nbr[k, 3]], new_s,
                                      empty_index, P, FWD0)
            Ctot_new += t0; Csp_new += s0
            t0, s0 = _bond_counts_dir(lat[nbr[k, 1]], new_s,
                                      empty_index, P, FWD1)
            Ctot_new += t0; Csp_new += s0
            t0, s0 = _bond_counts_dir(lat[nbr[k, 2]], new_s,
                                      empty_index, P, FWD2)
            Ctot_new += t0; Csp_new += s0

            # local energy delta
            dE_bonds = 0.0
            for t in range(m):
                kk = buf7[t]
                v  = _local_energy_site_hex(kk, lat, nbr, Bdir)
                newE[t]   = v
                dE_bonds += (v - oldE[t])

            dmu  = mu[new_s] - mu[old_s]
            dPhi = dE_bonds + dmu

            if (dPhi <= 0.0) or (np.random.rand() < np.exp(-beta * dPhi)):
                accepted = True

                for t in range(m):
                    e_sites[buf7[t]] = newE[t]
                energy += dE_bonds + dmu

                counts[old_s] -= 1
                counts[new_s] += 1

                if old_s == empty_index and new_s != empty_index:
                    Nocc += 1
                elif old_s != empty_index and new_s == empty_index:
                    Nocc -= 1

                Nmu  += ((1 if mask_mu[new_s] != 0 else 0)
                       - (1 if mask_mu[old_s] != 0 else 0))
                N_NS += ((1 if mask_ns[new_s] != 0 else 0)
                       - (1 if mask_ns[old_s] != 0 else 0))
                N_EW += ((1 if mask_ew[new_s] != 0 else 0)
                       - (1 if mask_ew[old_s] != 0 else 0))

                C_tot += (Ctot_new - Ctot_old)
                C_sp  += (Csp_new  - Csp_old)
                C_ns   = C_tot - C_sp
            else:
                lat[k] = old_s
                for t in range(m):
                    e_sites[buf7[t]] = oldE[t]

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


# -------------------- ring-buffer rollout --------------------------- #

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


# -------------------- public entrypoint ----------------------------- #

def simulate_lattice_triangular(
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
    """Monte Carlo single-site identity updates on a 2D periodic
    *triangular* lattice with 6-patch particles.

    Parameters
    ----------
    lattice_int        : (Ly, Lx) int species ids
    J_patch            : (n_patch, n_patch) energy lookup by patch ids
    patches_by_species : (n_species, 6) patch ids per species
                         in [E, NE, NW, W, SW, SE]
    mu_table           : (n_species,) chemical potentials
    empty_index        : species id for vacancies
    species_mu_indices : species ids counted in N_mu (for reweighting)
    eps_ns0, eps_sp0   : scalars for reporting
                         u = -(C_ns*eps_ns0 + C_sp*eps_sp0)/N_sites
    ns_species_ids, ew_species_ids : define Q = (N_NS - N_EW)/N_sites

    Returns
    -------
    snap_counts, energies, densities, lat_final
    """
    lattice_int = np.asarray(lattice_int, dtype=np.int64)
    P  = np.asarray(patches_by_species, dtype=np.int16)
    mu = np.asarray(mu_table, dtype=np.float64)

    if lattice_int.min() < 0 or lattice_int.max() >= P.shape[0]:
        raise ValueError("lattice_int has species ids outside patches_by_species")
    if mu.shape[0] != P.shape[0]:
        raise ValueError("mu_table length must equal n_species")
    if not (0 <= int(empty_index) < P.shape[0]):
        raise ValueError("empty_index out of range")
    if P.shape[1] != 6:
        raise ValueError("patches_by_species must have 6 columns for triangular lattice")
    if int(P.max(initial=0)) >= np.asarray(J_patch).shape[0]:
        raise ValueError("J_patch too small for patch ids in patches_by_species")

    Bdir = precompute_bond_table_hex(J_patch, P)

    n_species = P.shape[0]
    mask_mu = make_mask(n_species, species_mu_indices)
    mask_ns = make_mask(n_species, ns_species_ids)
    mask_ew = make_mask(n_species, ew_species_ids)

    (energies_buf, densities_buf,
     N_mu_buf, Nocc_buf, C_sp_buf, C_ns_buf, u_buf, Q_buf,
     sample_count, n_keep, acc_blocks,
     tries, accepts,
     lat_final) = _mc_triangular_single_site(
        lattice_int, Bdir, P, mu,
        mask_mu, mask_ns, mask_ew,
        int(empty_index),
        float(eps_ns0), float(eps_sp0),
        int(simulation_steps), float(beta),
        int(snapshot_interval), int(buffer_size),
        int(block_size),
    )

    energies  = _rollout_ring_1d(energies_buf,  sample_count)
    densities = _rollout_ring_2d(densities_buf, sample_count)

    N_sites   = int(lattice_int.size)
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
