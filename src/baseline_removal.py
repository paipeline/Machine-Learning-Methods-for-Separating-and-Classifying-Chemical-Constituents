import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

#use Asymmetric Least Squares (ALS) to remove the baseline
def baseline_als(y, lam, p, niter=10):
    length = len(y)
    estimate = sparse.diags([1,-2,1],[0,-1,-2], shape=(length,length-2))
    weight = np.ones(length)
    for i in range(niter):
        weight_matrix = sparse.spdiags(weight, 0, length, length)
        Z = weight_matrix + lam * estimate.dot(estimate.transpose())
        z = spsolve(Z, weight*y)
        weight = p * (y > z) + (1 - p) * (y < z)
    return z

def baseline_correct_compressed_data(compressed, lam=10^5, p=0.01, niter=10):
    corrected = {}
    for col, tuples in compressed.items():
        max_values = np.array([t[1] for t in tuples])
        corrected_max_values = max_values - baseline_als(max_values, lam, p, niter)
        corrected_tuples = [(tuples[i][0], corrected_max_values[i]) for i in range(len(tuples))]
        corrected[col] = corrected_tuples

    return corrected

