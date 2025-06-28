import numpy as np
from scipy.optimize import nnls

def bcd_solve(
    X,
    condition_matrix,
    max_iter=50,
    tol=1e-3,
    random_state=42,
    verbose=True
):
    """
    Block Coordinate Descent for a model with c cells, k topics (rank=1 each), no shared component.

    Model:
        X_{c x m} ≈ ∑_{j=1..k} [ condition_matrix[i,j] * H^j[i] ] A^j

    where:
      - X: (c x m) data matrix (cells x genes/features).
           * c = number of cells (rows).
           * m = number of genes/features (columns).
      - condition_matrix: (c x k) binary (0/1) matrix,
           condition_matrix[i,j] = 1 if topic j applies (is allowed) for cell i,
           else 0.
      - H^j: usage vector for topic j, shape (c, 1) => H^j[i,0] is how strongly topic j
           is used by cell i.
      - A^j: signature row for topic j, shape (1, m) => A^j[0,:] is the distribution of
           topic j over the m features.
      - We fix rank=1 per topic j, so the contribution of topic j to row i is:
           condition_matrix[i,j] * H^j[i] * A^j.

    The algorithm is a block‐coordinate descent:
      - In each iteration:
         (1) Compute the full reconstruction R_full = sum over all topics j
         (2) For each topic j in [1..k]:
             * "remove" topic j's current contribution from R_full to get a partial residual
             * update (H^j, A^j) by alternating NNLS (nonnegative least squares)
             * re-add the updated contribution to R_full
      - Check the change in objective function:  0.5 * || X - R_full ||^2_F
        If small enough, stop.

    Parameters
    ----------
    X : np.ndarray, shape (c, m)
        The data matrix: c = #cells, m = #genes/features.
    condition_matrix : np.ndarray, shape (c, k)
        0/1 matrix indicating if topic j is allowed for cell i.
    max_iter : int
        Maximum number of outer BCD iterations.
    tol : float
        Convergence threshold on change in objective.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        If True, prints progress every few iterations.

    Returns
    -------
    H_list : list of length k
        H_list[j] => a (c,1) array for topic j's usage across c cells.
    A_list : list of length k
        A_list[j] => a (1,m) array for topic j's distribution over m features.

    Notes
    -----
    - Because rank=1, each topic j is effectively an outer product: H^j * A^j.
    - We store H^j in shape (c,1) and A^j in shape (1,m).
    - We use 'nnls' from scipy to ensure nonnegativity.
    """

    # --------------------------------------------------------------------
    # 0) Initialization and setup
    # --------------------------------------------------------------------
    np.random.seed(random_state)       # fix seed for reproducibility
    c, m = X.shape                     # c = #cells, m = #genes
    k = condition_matrix.shape[1]      # k = #topics

    # Create lists to hold (H^j, A^j) for j in [0..k-1]
    # H^j => shape (c,1),   A^j => shape (1,m).
    H_list = [np.random.rand(c, 1) for _ in range(k)]
    A_list = [np.random.rand(1, m) for _ in range(k)]

    # --------------------------------------------------------------------
    # 1) Full reconstruction:  sum_j [ (cond(i,j)*H^j[i]) * A^j ]
    #    i.e. for each topic j, we compute (masked_H^j) @ A^j, then sum
    # --------------------------------------------------------------------
    def full_reconstruction(H_list, A_list, cond_mat):
        """
        Returns an (c x m) matrix = sum_j( masked_Hj @ Aj ),
        where masked_Hj[i,0] = cond_mat[i,j]*H_list[j][i,0].
        """
        R_total = np.zeros(X.shape, dtype=float)  # shape (c x m)
        for j_topic in range(k):
            # shape(H_list[j_topic]) = (c,1)
            # masked_Hj => multiply each row i by cond_mat[i,j_topic]
            masked_Hj = cond_mat[:, j_topic:j_topic+1] * H_list[j_topic]
            # Now (c,1) @ (1,m) => (c,m)
            R_j = masked_Hj @ A_list[j_topic]
            R_total += R_j
        return R_total

    # --------------------------------------------------------------------
    # 2) Objective = 0.5 * || X - R ||^2_F
    # --------------------------------------------------------------------
    def objective(X, H_list, A_list, cond_mat):
        R = full_reconstruction(H_list, A_list, cond_mat)
        return 0.5 * np.sum((X - R)**2)

    # --------------------------------------------------------------------
    # 3) Updating topic j => (H^j, A^j) with partial residual
    # --------------------------------------------------------------------
    def update_topic_j(X, Hj, Aj, cond_j, partial_resid):
        """
        partial_resid: shape (c x m), equals X - sum_{l != j} ...
        so we want partial_resid ~ (cond_j * Hj) @ Aj.

        Because rank=1:
          - Hj is shape (c,1)
          - Aj is shape (1,m)

        We'll do an alternating approach:
          (a) fix Aj => solve for Hj row-by-row, NNLS
          (b) fix Hj => solve for Aj col-by-col, NNLS

        Each sub-step is effectively 1D linear regression with nonnegativity.
        """
        c_cells, m_genes = X.shape

        # --- (a) Update Hj row by row ---
        # For cell i, partial_resid[i,:] ~ cond_j[i]*Hj[i,0]*Aj[0,:].
        # We want to solve for scale = Hj[i,0].
        # 1D formula => scale = max(0, (b dot Aj[0,:]) / (Aj[0,:] dot Aj[0,:]))
        Aj_row = Aj[0, :]  # shape (m,)
        for i in range(c_cells):
            if cond_j[i] == 0:
                # If topic j is disallowed for cell i, force usage=0
                Hj[i, 0] = 0.0
            else:
                b_i = partial_resid[i, :]  # shape (m,)
                denom = np.dot(Aj_row, Aj_row)
                numer = np.dot(b_i, Aj_row)
                scale = 0.0
                if denom > 1e-12 and numer > 0.0:
                    scale = numer / denom
                Hj[i, 0] = scale

        # --- (b) Update Aj column by column ---
        # partial_resid[:, col] ~ (cond_j * Hj[:,0]) * Aj[0,col].
        # We again do a 1D solve => scale_j = max(0, (b dot M) / (M dot M)).
        M = cond_j[:, None] * Hj  # shape (c,1)
        for col in range(m_genes):
            b_c = partial_resid[:, col]  # shape (c,)
            denom = np.dot(M[:,0], M[:,0])
            numer = np.dot(b_c, M[:,0])
            scale = 0.0
            if denom > 1e-12 and numer > 0.0:
                scale = numer / denom
            Aj[0, col] = scale

        return Hj, Aj

    # --------------------------------------------------------------------
    # 4) Main BCD loop
    # --------------------------------------------------------------------
    old_obj = None
    for it in range(max_iter):
        # Compute the full reconstruction with current factors
        R_full = full_reconstruction(H_list, A_list, condition_matrix)

        # --- For each topic j, isolate partial residual, update (H^j, A^j) ---
        for j_topic in range(k):
            # old_contrib = (cond_j * H^j) @ A^j
            masked_Hj = condition_matrix[:, j_topic:j_topic+1] * H_list[j_topic]
            old_contrib = masked_Hj @ A_list[j_topic]

            # partial_resid = X - (R_full - old_contrib)
            # i.e. the difference ignoring j_topic's old contribution
            partial_resid = X - (R_full - old_contrib)

            # Update that topic
            H_list[j_topic], A_list[j_topic] = update_topic_j(
                X,
                H_list[j_topic],
                A_list[j_topic],
                condition_matrix[:, j_topic],
                partial_resid
            )

            # Re-add j_topic's new contribution to R_full so next topic sees updated sum
            new_contrib = (condition_matrix[:, j_topic:j_topic+1] * H_list[j_topic]) @ A_list[j_topic]
            R_full = (R_full - old_contrib) + new_contrib

        # -- Check objective & possibly stop if small change --
        cur_obj = objective(X, H_list, A_list, condition_matrix)
        if old_obj is not None and abs(old_obj - cur_obj) < tol:
            if verbose:
                print(f"BCD converged at iteration {it}, objective={cur_obj:.6f}")
            break
        old_obj = cur_obj

        if verbose and ((it+1) % 5 == 0):
            print(f"Iteration {it+1}, objective={cur_obj:.6f}")

    return H_list, A_list

def merge_factors(H_list, A_list, condition_matrix, epsilon=1e-12):
    """
    Convert (H^j, A^j) => (theta, beta).

    Model:
      X_{c x m} ≈ ∑_{j=1..k} [ condition_matrix[i,j]*H^j[i] ] A^j.

    Final usage:
      theta[i,j] = H^j[i], ( but 0 if condition_matrix[i,j]==0 )
    Final signature:
      beta[j,:]  = A^j[0,:]   ( shape (1,m) => flatten to row vector )

    We optionally row-normalize 'theta' so each row i sums to 1,
    and row-normalize 'beta' so each row j sums to 1, if we want
    them to represent probability distributions (like LDA).
    """

    c, k = condition_matrix.shape
    # A_list[j].shape => (1,m)
    # H_list[j].shape => (c,1)

    # 1) Build theta
    theta = np.zeros((c, k))
    for j in range(k):
        # usage_j: shape (c,) from H_list[j][:, 0]
        usage_j = H_list[j].ravel()  # (c,)
        # multiply by condition_matrix to enforce zero where cond=0
        usage_j *= condition_matrix[:, j]
        theta[:, j] = usage_j

    # 2) Build beta
    # row j => A_list[j][0,:]
    m = A_list[0].shape[1]
    beta = np.zeros((k, m))
    for j in range(k):
        beta[j, :] = A_list[j][0, :]

    # 3) Normalize rows of theta, rows of beta
    #    typical if we want them as probability distributions
    row_sums_theta = np.sum(theta, axis=1, keepdims=True) + epsilon
    theta /= row_sums_theta

    row_sums_beta = np.sum(beta, axis=1, keepdims=True) + epsilon
    beta /= row_sums_beta

    return theta, beta

def create_condition_matrix(
    cell_labels,
    topic_hierarchy
):
    """
    Build a GMDF condition matrix from a topic hierarchy and each cell's labeled cell_type.
    Ensures that each cell has 1's *only* for the node IDs in its path.

    Parameters
    ----------
    cell_labels : list or np.ndarray of strings
        Each element is the "leaf label" (cell type).
        Must be one of the keys in topic_hierarchy.
        Length == number of cells (rows).
    topic_hierarchy : dict
        Maps each leaf label (e.g. "CD141-positive myeloid dendritic cell")
        to a list of node IDs (e.g. [0,1,3]).

    Returns
    -------
    condition_matrix : np.ndarray (int)
        Shape (n_cells x p), where p = total number of unique node IDs.
        Row i is 1 for each node in cell_labels[i]’s path, else 0.
    sorted_node_ids : list of int
        The sorted list of all unique node IDs used as columns in the matrix.
        So column j in condition_matrix corresponds to sorted_node_ids[j].
    """

    n_cells = len(cell_labels)

    # 1) Gather all unique node IDs
    all_node_ids = set()
    for path in topic_hierarchy.values():
        all_node_ids.update(path)

    # 2) Sort them
    sorted_node_ids = sorted(all_node_ids)
    p = len(sorted_node_ids)

    # 3) Build a dict: node_id -> column index
    nodeid2col = {nid: idx for idx, nid in enumerate(sorted_node_ids)}

    # 4) Initialize (n_cells x p) matrix of zeros
    condition_matrix = np.zeros((n_cells, p), dtype=int)

    # 5) For each cell, set 1's for each node in that leaf’s path
    for i, leaf_label in enumerate(cell_labels):
        if leaf_label not in topic_hierarchy:
            raise ValueError(f"Cell type '{leaf_label}' not in topic_hierarchy keys!")

        path_node_ids = topic_hierarchy[leaf_label]
        for node_id in path_node_ids:
            col_j = nodeid2col[node_id]
            condition_matrix[i, col_j] = 1

    return condition_matrix, sorted_node_ids

def gmdf_infer_test(
    E_test,
    condition_matrix_test,
    A_list_fixed,
    max_iter=20,
    tol=1e-3,
    random_state=42
):
    """
    Infer new usage H_list_test for held-out cells, given fixed topic signatures A_list_fixed.

    - E_test: shape (n_test x m)
    - condition_matrix_test: (n_test x k) binary
    - A_list_fixed: list of length k, each shape (1 x m) => learned from training
    - Return: H_list_test => list of length k, each shape (n_test x 1)

    We'll do a small block-coordinate approach:
      1) Initialize H_list_test randomly
      2) Repeatedly loop over topics j:
         - partial_resid = E_test - sum_{l != j}( masked_Hl_test @ A_l )
         - update H^j_test row-by-row via 1D NNLS
    """
    np.random.seed(random_state)
    n_test, m = E_test.shape
    k_topics = len(A_list_fixed)

    # 1) Initialize usage
    H_list_test = [np.random.rand(n_test, 1) for _ in range(k_topics)]

    for it in range(max_iter):
        # a) Compute full reconstruction = sum_j( masked_Hj @ A_j )
        R_full = np.zeros(E_test.shape, dtype=float)
        for j in range(k_topics):
            masked_Hj = condition_matrix_test[:, j:j+1] * H_list_test[j]
            R_full += masked_Hj @ A_list_fixed[j]  # => shape (n_test x m)

        # b) For each topic j, isolate partial_resid & update H_list_test[j]
        changed = 0.0  # track how much usage changes, for optional convergence check
        for j in range(k_topics):
            # old contribution
            old_contrib = (condition_matrix_test[:, j:j+1] * H_list_test[j]) @ A_list_fixed[j]

            partial_resid = E_test - (R_full - old_contrib)
            
            # Update usage for topic j row-by-row
            A_j = A_list_fixed[j][0, :]  # shape (m,) 
            for i in range(n_test):
                if condition_matrix_test[i, j] == 0:
                    # not allowed
                    if H_list_test[j][i,0] != 0.0:
                        H_list_test[j][i,0] = 0.0
                else:
                    b_i = partial_resid[i,:]  # shape (m,)
                    denom = np.dot(A_j, A_j)
                    numer = np.dot(b_i, A_j)
                    new_val = 0.0
                    if denom > 1e-12 and numer > 0.0:
                        new_val = numer / denom
                    old_val = H_list_test[j][i,0]
                    changed += abs(new_val - old_val)
                    H_list_test[j][i,0] = new_val

            # Re‐add updated jth contribution
            new_contrib = (condition_matrix_test[:, j:j+1] * H_list_test[j]) @ A_list_fixed[j]
            R_full = R_full - old_contrib + new_contrib

        # optional: if changed < tol => break
        # pass

    return H_list_test

def merge_gmdf_test_factors(H_list_test, A_list_fixed, cond_test, epsilon=1e-12):
    """
    Convert the new usage H_list_test + fixed A_list => final test usage theta_test.
    We skip building a 'beta_test' because the topics are from training.

    Returns
    -------
    theta_test: shape (n_test, k)
    """
    n_test, k = cond_test.shape
    theta_test = np.zeros((n_test, k))
    for j in range(k):
        usage_j = H_list_test[j].ravel()
        usage_j *= cond_test[:, j]
        theta_test[:, j] = usage_j

    # row-normalize
    row_sums = theta_test.sum(axis=1, keepdims=True) + epsilon
    theta_test /= row_sums

    return theta_test
