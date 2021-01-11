import torch
import numpy as np
from qrsolver import solve

# from scipy.sparse.linalg import spsolve
from scipy.sparse import random, coo_matrix

device = torch.device("cuda")
# def cuda_solve


def random_system(size, density):
    result = random(size, size, density=0.5)
    row = result.row
    col = result.col
    data = result.data

    # sort by row major. Yes, the order is flipped, read the docs
    order = np.lexsort((col, row))
    row = row[order]
    col = col[order]
    data = data[order]

    A = coo_matrix((data, (row, col)), shape=(size, size)).toarray()
    # generate x
    x = np.random.rand(size)
    b = np.matmul(A, x)

    row = torch.from_numpy(row).to(device)
    col = torch.from_numpy(col).to(device)
    data = torch.from_numpy(data).to(device).double()
    A = torch.from_numpy(A).to(device).double()
    x = torch.from_numpy(x).to(device).double()
    b = torch.from_numpy(b).to(device).double()

    is_singular = A.det() == 0
    return (row, col, data, A, x, b, is_singular)


def qr_solve(size, row, col, data, b, tol=1e-10):
    nnz = data.size(0)
    dcsrRow = torch.empty(size + 1, dtype=torch.int, device=device)
    dx = torch.empty(size, dtype=torch.double, device=device)
    singularity = solve(nnz, size, tol, data, col, row, dcsrRow, b, dx)
    if singularity != -1:
        print("Error: 6CBP3")
        print("singular matrix")
        __import__("pdb").set_trace()
    return dx


# final_connect_indices, inverse = torch.unique(
# connect_indices, sorted=True, dim=0, return_inverse=True
# )

if __name__ == "__main__":
    size = 5
    while True:
        row, col, data, A, x, b, is_singular = random_system(size, 0.5)
        if not is_singular:
            break
    dx = qr_solve(size, row, col, data, b)
    error = torch.sum(torch.abs(x - dx))
    print(error)
