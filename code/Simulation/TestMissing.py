import numpy as np
import time

def T_M(Z, M):
    """
    Computes the test statistic T_M(Z, M) = sum_i sum_j { Z_ij * (sum_k M_ijk) }

    Parameters:
        Z: array_like, shape (N, 1)
            Treatment assignment vector
        M: array_like, shape (N, K)
            Missingness indicator matrix (1 = missing, 0 = observed)

    Returns:
        T: float
            The computed test statistic
    """
    Z = np.asarray(Z).reshape(-1, 1)     # (N, 1)
    M = np.asarray(M)                    # (N, K)
    return np.sum(Z * np.sum(M, axis=1).reshape(-1, 1))

def getZsimTemplates(Z_sorted, S):
    Z_sim_templates = []
    unique_strata = np.unique(S)
    for stratum in unique_strata:
        indices = np.where(S == stratum)[0]
        p = np.mean(Z_sorted[indices])
        size = len(indices)
        template = [0] * int(size * (1 - p)) + [1] * int(size * p)
        Z_sim_templates.append(template)
    return Z_sim_templates

def getZsim(Z_sim_templates):
    Z_sim = []
    for tpl in Z_sim_templates:
        arr = np.array(tpl.copy())
        np.random.shuffle(arr)
        Z_sim.append(arr)
    return np.concatenate(Z_sim).reshape(-1, 1)

def test_missing(Z, M, S=None, L=1000, randomization_design='strata',
                 verbose=False, alternative="greater", alpha=0.05, random_state=None):
    """
    Randomization test for testing association between treatment assignment Z
    and missingness pattern M using the test statistic T_M(Z, M)

    Parameters:
        Z: array_like, shape (N, 1)
            Treatment assignment vector
        M: array_like, shape (N, K)
            Missingness indicator matrix
        S: array_like, shape (N,), optional
            Strata or cluster labels (for stratified/clustered randomization)
        L: int
            Number of randomization iterations
        ...

    Returns:
        reject: bool
        p_value: float
    """
    start_time = time.time()

    t_obs = T_M(Z, M)
    if verbose:
        print(f"Observed T_M(Z, M): {t_obs:.4f}")

    if randomization_design == 'strata':
        Z_sim_templates = getZsimTemplates(Z, S)
    else:
        p = 0.5
        cluster_ids = np.unique(S)
        num_clusters = len(cluster_ids)
        cluster_template = np.array([0] * int(num_clusters * (1 - p)) + [1] * int(num_clusters * p))

    t_sim = np.zeros(L)
    for l in range(L):
        if randomization_design == 'strata':
            Z_sim = getZsim(Z_sim_templates)
        else:
            np.random.shuffle(cluster_template)
            Z_sim = np.array([cluster_template[int(s) - 1] for s in S.flatten()]).reshape(-1, 1)

        t_sim[l] = T_M(Z_sim, M)

        if verbose and l % max(1, L // 10) == 0:
            print(f"Simulation {l+1}/{L}: T_sim = {t_sim[l]:.4f}")

    if alternative == "greater":
        p_value = np.mean(t_sim >= t_obs)
    elif alternative == "less":
        p_value = np.mean(t_sim <= t_obs)
    else:  # two-sided
        p_value = np.mean(np.abs(t_sim - np.mean(t_sim)) >= np.abs(t_obs - np.mean(t_sim)))

    reject = p_value < alpha

    if verbose:
        print(f"\nP-value: {p_value:.4f} | Reject H0: {reject}")
        print(f"Total time: {time.time() - start_time:.2f} seconds")

    return reject, p_value
