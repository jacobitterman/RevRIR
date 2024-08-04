import numpy as np
from collections import Counter

def get_sim_mat(indices=np.array([1, 2, 3, 2, 4, 1, 5, 6, 3, 4]), verbose=False):
    # Find indices with multiple occurrences
    unique_indices, inverse_inds, counts = np.unique(indices, return_counts=True, return_inverse=True)

    matrix = np.zeros((len(indices), len(indices)), dtype=int)
    for ind in unique_indices:
        i = np.where(ind == indices)[0]
        for i_ in i:
            for j_ in i:
                matrix[i_, j_] = 1

    if verbose:
        print("Original indices:")
        print(indices)

        print("\nMatrix showing occurrences:")
        print(matrix)
    return matrix


def get_sim_mat_and_min_exp_loss(indices):
    m = get_sim_mat(indices)
    labels = m / np.sum(m, axis=1)

    c = Counter(1 /  np.diag(labels))
    min_loss = 0
    for k, v in c.items():
        min_loss += -v / labels.shape[0] * np.log(1 / k)
    return labels, min_loss
