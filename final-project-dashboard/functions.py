import numpy as np
import numpy as np

def rref(A, tol=1e-12, ignore_last=False):
    A = A.astype(float).copy()
    pivots = []
    row = 0
    m, n = A.shape

    for c in range(n):
        if ignore_last and c == n - 1:
            continue

        # find pivot row
        pivot = None
        for r in range(row, m):
            if abs(A[r, c]) > tol:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]

        A[row] = A[row] / A[row, c]

        for r in range(m):
            if r != row:
                A[r] -= A[row] * A[r, c]

        pivots.append(c)   
        row += 1
        if row == m:      
            break

    A[np.abs(A) < tol] = 0.0
    return A, np.array(pivots)


def build_nullspace(R, pivots, tol=1e-12):
    R = R.astype(float)
    m, n = R.shape
    pivots = list(pivots)

    free_cols = [j for j in range(n) if j not in pivots]
    f = len(free_cols)

    if f == 0:
        return np.zeros((n, 0))

    N = np.zeros((n, f))
    for k, free_j in enumerate(free_cols):
        x = np.zeros(n)
        x[free_j] = 1.0
        for i, pivot_j in enumerate(pivots):
            x[pivot_j] = -R[i, free_j]
        N[:, k] = x

    N[np.abs(N) < tol] = 0.0
    return N


def compute_solution(A, b, tol=1e-12):
    A = A.astype(float)
    b = b.astype(float)

    m, n = A.shape
    Aug = np.column_stack([A, b])

    R_aug, pivots = rref(Aug, tol=tol, ignore_last=True)
    R = R_aug[:, :n]
    rhs = R_aug[:, n]

    for i in range(m):
        if np.all(np.abs(R[i]) < tol) and abs(rhs[i]) > tol:
            return np.zeros((n, 0)), "none"

    x_p = np.zeros(n)
    for i, pcol in enumerate(pivots):
        x_p[pcol] = rhs[i]

    N = build_nullspace(R, pivots, tol)

    if N.shape[1] == 0:
        return x_p, "unique"
    else:
        return np.column_stack([x_p, N]), "infinite"
    
def least_squares(A, b):
    b_hat = A.T @ b
    A_square = A.T @ A
    sol, status = compute_solution(A_square, b_hat)
    if status == "none":
        raise ValueError("Normal equations reported inconsistent (should not happen in exact arithmetic).")
    if status == "unique":
        x_hat = sol
    else: 
        x_hat = sol[:, 0] 
    return x_hat

def compute_determinant(A, tol=1e-12):
    det = 1
    row = 0
    A = A.copy()
    for c in range(len(A[0])):
        sign = 1
        #check which column has the pivot
        pivot = None
        for r in range(row, len(A)):
            if abs(A[r, c]) > tol:
                pivot = r
                break
        if pivot is None:
            det *= 0
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            sign *= -1
        det *= sign * A[row][c]
        for r in range(len(A)):
            if r != row and A[r][c] != 0:
                A[r] -= A[row] * (A[r][c]/A[row][c])
        row += 1
        if row == len(A[0]):
            break
    return det

def least_squares(A, b):
    b_hat = A.T @ b
    A_square = A.T @ A
    sol, status = compute_solution(A_square, b_hat)
    if status == "none":
        raise ValueError("Normal equations reported inconsistent (should not happen in exact arithmetic).")
    if status == "unique":
        x_hat = sol
    else: 
        x_hat = sol[:, 0] 
    return x_hat

def compute_svd(A, tol=1e-12):
    A = A.astype(float)

    ATA = A.T @ A
    eigvals, V = np.linalg.eigh(ATA)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    sing = np.sqrt(np.clip(eigvals, 0, None))

    r = np.sum(sing > tol)
    sing_r = sing[:r]
    V_r = V[:, :r]

    U_r = (A @ V_r) / sing_r

    Sigma_r = np.diag(sing_r)

    return U_r, Sigma_r, V_r.T

def A_n(S, x, n):
    X = np.diag(x**n)
    return S @ X @ np.linalg.inv(S)

def generate_spd(n=3, eps=1e-6, seed=None):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    A = B.T @ B + eps * np.eye(n)
    return A

