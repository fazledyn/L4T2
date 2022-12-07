import numpy as np


RANGE = 20
def main():
    
    n, m = input("Enter dimension(n, m): ").split(",")
    n, m = int(n.strip()), int(m.strip())
    
    mat = np.random.randint(-RANGE, RANGE, size=(n, m))
    # print(f"{mat=}\n")

    U, D, VT = np.linalg.svd(mat)
    # print(f"U:\n{U}\nD:\n{D}\nVT:\n{VT}")

    #   U -> n x n 
    #   D -> min x min
    #   VT -> m x m

    UT = U.T
    V = VT.T

    D_temp = np.diag(D)
    # print(f"{D_temp=}\n")

    D_temp = np.reciprocal(D_temp, where=(D_temp!=0))
    # print(f"{D_temp=}\n")

    D_plus = np.zeros(shape=(n, m))

    for i in range(len(D_temp)):
        D_plus[i][i] = D_temp[i][i]

    # print(f"{D_plus=}\n")

    D_plus = np.transpose(D_plus)
    # print(f"{D_plus=}\n")

    A_plus = np.matmul(np.matmul(V, D_plus), UT)
    # print(f"{A_plus=}\n")

    A_np = np.linalg.pinv(mat)
    # print(f"{A_np=}\n")

    if np.allclose(A_plus, A_np):
        print("Inverses are equal!")
    else:
        print("Inverses aren't equal!")


if __name__ == "__main__":
    main()

