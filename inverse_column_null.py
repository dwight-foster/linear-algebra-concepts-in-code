"""
inverse_column_null.py

This script provides functions to compute the inverse of a matrix (if it exists), the column space, and the null space of a matrix using NumPy and SciPy.
"""

import numpy as np
from scipy.linalg import null_space


def matrix_inverse(A):
    """
    Returns the inverse of matrix A if it exists, otherwise raises a LinAlgError.
    """
    return np.linalg.inv(A)


def column_space(A, tol=1e-10):
    """
    Returns an orthonormal basis for the column space of matrix A.
    """
    # Use QR decomposition to get the basis
    Q, R = np.linalg.qr(A)
    # Only keep columns of Q corresponding to nonzero rows in R
    rank = np.sum(np.abs(np.diag(R)) > tol)
    return Q[:, :rank]


def nullspace(A, tol=1e-10):
    """
    Returns an orthonormal basis for the null space of matrix A.
    """
    return null_space(A, rcond=tol)


if __name__ == "__main__":
    # Example usage
    A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=float)
    print("Matrix A:\n", A)
    try:
        invA = matrix_inverse(A)
        print("\nInverse of A:\n", invA)
    except np.linalg.LinAlgError:
        print("\nMatrix A is singular and does not have an inverse.")

    print("\nColumn space of A (orthonormal basis):\n", column_space(A))
    print("\nNull space of A (orthonormal basis):\n", nullspace(A))
