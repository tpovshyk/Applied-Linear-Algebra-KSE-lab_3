import numpy as np


def svd (A):
    ATA = np.dot(A.T, A)
    eigvals_v, V = np.linalg.eigh(ATA)

    sorted_indices = np.argsort(eigvals_v)[::-1]
    eigvals_v = eigvals_v[sorted_indices]
    V = V[:, sorted_indices]

    singular_values = np.sqrt(eigvals_v)

    U = np.dot(A, V)
    for i in range(U.shape[1]):
        U[:, i] /= singular_values[i]

    Sigma = np.zeros(A.shape)
    np.fill_diagonal(Sigma, singular_values)

    return U, Sigma, V.T

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
U, Sigma, VT = svd(A)
A_reconstructed = np.dot(U, np.dot(Sigma, VT))

print("Original matrix:\n", A)
print("Reconstructed matrix:\n", A_reconstructed)
print("U:\n", U)
print("Sigma:\n", Sigma)
print("V.T:\n", VT)
