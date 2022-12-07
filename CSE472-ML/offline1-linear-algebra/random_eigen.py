import numpy as np


RANGE = 20
def main():
    
    n = input("Enter dimension(n): ")
    n = int(n)

    mat = np.random.randint(-RANGE, RANGE, size=(n, n))
    while np.linalg.det(mat) == 0:
        mat = np.random.randint(-RANGE, RANGE, size=(n, n))

    print("Matrix:\n", mat)

    eig_val, eig_vec = np.linalg.eig(mat)
    inv_eig_vec = np.linalg.inv(eig_vec)
    A = np.diag(eig_val)
    
    re_mat = np.dot(eig_vec, np.dot(A, inv_eig_vec))

    if np.allclose(mat, re_mat):
        print("Reconstruction is perfect!")
    else:
        print("Reconstruction is not perfect!")


if __name__ == "__main__":
    main()

