# Flaky Clustering Library v0.1
# Big MSSC (Minimum Sum-Of-Squares Clustering)

# Nenad Mladenovic, Rustam Mussabayev, Alexander Krassovitskiy
# rmusab@gmail.com


# v0.1  - 30/06/2021 - Bug fixing in multi_portion_mssc
# v0.09 - 10/11/2020 - Revision of shake_centers logic
# v0.08 - 18/09/2020 - Bug fixing;
# v0.07 - 19/07/2020 - New functionality: distance matrices calculation routines with GPU support; different distance metrics; revision of optimal number of clusters routine;
# v0.06 - 05/06/2020 - New functionality: method sequencing;
# v0.05 - 04/06/2020 - New functionality:  Simple center shaking VNS, Membership shaking VNS, Iterative extra center insertion/deletion, procedure for choosing the new n additional centers for existing ones using the k-means++ logic;
# v0.04 - 17/03/2020 - Different initialization modes were added to "Decomposition/aggregation k-means";
# v0.03 - 13/03/2020 - New functionality: k-means++;
# v0.02 - 10/03/2020 - New functionality: Decomposition/aggregation k-means;
# v0.01 - 27/02/2020 - Initial release: multiprocessing k-means.

import math
import time
import pickle
import threading
import cupy as cp
import numpy as np
import numba as nb
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from numba import njit, prange, objmode, cuda


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
def save_obj(obj, name):
    pickle.dump(obj,open(name + '.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    
def normalization(X):
    X_min = np.amin(X, axis=0)
    X = X - X_min
    X_max = np.amax(X, axis=0)
    if X_max.ndim == 1:
        X_max[X_max == 0.0] = 1.0
    elif X_max.ndim == 0:
        if X_max == 0.0:
            X_max = 1.0
    else:
        X_max = 1.0
    X = X / X_max
    return X


@njit(parallel=True)
def normalization1D(X, min_scaling = True):
    assert X.ndim == 1
    n = X.shape[0]
    X_min = np.inf
    for i in range(n):
        if X[i] < X_min:
            X_min = X[i]
    X_max = np.NINF
    if min_scaling:
        for i in range(n):
            X[i] -= X_min
            if X[i] > X_max:
                X_max = X[i]
    else:
        for i in range(n):
            if X[i] > X_max:
                X_max = X[i]        
    if X_max != 0:
        for i in prange(n):
            X[i] = X[i]/X_max
            
            
@njit(parallel=True)
def normalization2D(X, min_scaling = True):
    assert X.ndim == 2
    n, m = X.shape
    for i in prange(m):
        min_val = np.inf
        for j in range(n):
            if X[j,i] < min_val:
                min_val = X[j,i]
        max_val = np.NINF
        if min_scaling:
            for j in range(n):
                X[j,i] -= min_val
                if X[j,i] > max_val:
                    max_val = X[j,i]
        else:
            for j in range(n):
                if X[j,i] > max_val:
                    max_val = X[j,i]
        if max_val != 0.0:
            for j in range(n):
                X[j,i] = X[j,i]/max_val



# Generate isotropic Gaussian blobs
def gaussian_blobs(n_features = 2, n_samples = 1000, n_clusters = 5, cluster_std = 0.1):
    true_centers = np.random.rand(n_clusters, n_features)   
    X, labels = make_blobs(n_samples=n_samples, centers=true_centers, cluster_std=cluster_std)
    N = np.concatenate((true_centers,X))
    N = normalization(N)
    true_centers = N[:n_clusters]
    X = N[n_clusters:]
    return X, true_centers, labels


def draw_dataset(X, true_centers, original_labels, title = 'DataSet'):
    n_clusters = len(true_centers)
    plt.rcParams['figure.figsize'] = [10,10]
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    
    if original_labels.shape[0] == X.shape[0]:
        for k, col in zip(range(n_clusters), colors):
            my_members = original_labels == k

            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')

            if true_centers.shape[0] > 0:
                cluster_center = true_centers[k]
                plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    else:
        plt.plot(X[:, 0], X[:, 1], '.')
    
    plt.title('DataSet')
    plt.show()

    
def generate_grid_nodes(coordinates1D, n_dimensions=3):
    coordinatesND = n_dimensions*[coordinates1D]
    mesh = np.array(np.meshgrid(*coordinatesND))
    grid_nodes = mesh.T.reshape(-1, n_dimensions)
    return grid_nodes


def generate_blobs_on_grid(n_samples=3000, grid_size=3, n_features=3, standard_deviation = 0.1):
    assert grid_size > 0
    cell_size = 1/grid_size
    half_cell_size = cell_size/2
    coordinates1D = np.linspace(half_cell_size, 1.0-half_cell_size, grid_size)
    true_centroids = generate_grid_nodes(coordinates1D, n_features)    
    samples, sample_membership = make_blobs(n_samples=n_samples, centers=true_centroids, cluster_std=standard_deviation)
    mask = np.all((samples >= 0.0) & (samples <= 1.0) , axis = 1)
    samples = samples[mask]
    sample_membership = sample_membership[mask]    
    return samples, sample_membership, true_centroids

                    
@njit(inline='always')
def condensed_size(matrix_size):
    return int((matrix_size*(matrix_size-1))/2)


@njit(inline='always')
def condensed_idx(i,j,n):
    return int(i*n + j - i*(i+1)/2 - i - 1)


@njit(inline='always')
def regular_idx(condensed_idx, n):
    i = int(math.ceil((1/2.) * (- (-8*condensed_idx + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))
    ii = i+1
    j = int(n - (ii * (n - 1 - ii) + (ii*(ii + 1))/2) + condensed_idx)
    return i,j


@njit(parallel = True)
def row_of_condensed_matrix(row_ind, condensed_matrix):
    assert row_ind > -1
    condensed_len = len(condensed_matrix)
    size = matrix_size(condensed_matrix)
    assert row_ind < size
    out = np.empty(size, dtype = condensed_matrix.dtype)
    out[row_ind] = 0
    if row_ind < size-1:
        ind1 = condensed_idx(row_ind, row_ind+1, size)
        ind2 = condensed_idx(row_ind, size-1, size)
        out[row_ind+1:size] = condensed_matrix[ind1:ind2+1]
    for i in prange(0,row_ind):
        out[i] = condensed_matrix[condensed_idx(i,row_ind,size)]
    return out


@njit
def matrix_size(condensed_matrix):
    n = math.ceil((condensed_matrix.shape[0] * 2)**.5)
    if (condensed_matrix.ndim != 1) or (n * (n - 1) / 2 != condensed_matrix.shape[0]):
        raise ValueError('Incompatible vector size.')
    return n


@njit(inline='always')
def matrix_element(i, j, N, condensed_matrix, diagonal_value):
    if   j > i:
        return condensed_matrix[condensed_idx(i,j,N)]
    elif j < i:
        return condensed_matrix[condensed_idx(j,i,N)]
    else:
        return diagonal_value
    
    
# Extraction of submatrix from condensed_matrix where
# rows, cols - indices of rows and columns which must be included to submatrix
# diagonal_value - diagonal value in the original full square matrix
# Functionality is similar to Advanced Indexing in Numpy: submatrix = matrix[rows][:,cols]
def _submatrix(condensed_matrix, rows, cols, diagonal_value):
    N = matrix_size(condensed_matrix)    
    if (condensed_matrix.ndim != 1) or (rows.ndim != 1) or (cols.ndim != 1) or ((N * (N - 1) / 2) != condensed_matrix.shape[0]):
        raise ValueError('Incompatible vector size.')       
    if (N > 0) and (condensed_matrix.ndim == 1) and (rows.ndim == 1) and (cols.ndim == 1) and ((N * (N - 1) / 2) == condensed_matrix.shape[0]):
        if len(rows) == 0:
            new_rows = np.arange(N)
        else:
            new_rows = rows
        if len(cols) == 0:
            new_cols = np.arange(N)
        else:
            new_cols = cols           
        n_rows = len(new_rows)
        n_cols = len(new_cols)        
        chunk = np.empty((n_rows, n_cols), dtype = condensed_matrix.dtype)
        chunk = np.empty((n_rows, n_cols), dtype = condensed_matrix.dtype)
        for i in prange(n_rows):
            for j in range(n_cols):
                chunk[i,j] = matrix_element(new_rows[i],new_cols[j],N,condensed_matrix,diagonal_value)
    else:
        chunk = np.empty((0, 0), dtype = condensed_matrix.dtype)  
    return chunk


submatrix = njit(parallel=False)(_submatrix)
submatrix_parallel = njit(parallel=True)(_submatrix)


# Squared Euclidian distance (standard realization)
@njit(inline='always')
def euclidian2_distance(u, v):
    d = u.dtype.type(0.0)
    for i in range(u.shape[0]):
        d += (u[i] - v[i]) ** 2.0
    return d    


@njit(inline='always')
def cosine_distance(u, v):
    n = u.shape[0]
    udotv = 0.
    u_norm = 0.
    v_norm = 0.
    for i in range(n):
        udotv  += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]
    if (u_norm == 0.) or (v_norm == 0.):
        d = 1.
    else:
        d = abs(1.-udotv / (u_norm * v_norm) ** .5) # Return absolute value to avoid small negative value due to rounding
    return u.dtype.type(d)


@njit(inline='always')
def calc_distance(u, v, distance_measure=0):
    if distance_measure == 0:
        d = euclidian2_distance(u, v)
    else:
        d = cosine_distance(u, v)
    return d
    
    
@cuda.jit
def distance_matrix_gpu(X, out, distance_measure=0):
    i,j = cuda.grid(2)
    n = X.shape[0]    
    if (i < n) and (j < n):
        if j > i:
            d = calc_distance(X[i], X[j], distance_measure)
            out[i,j] = d
            out[j,i] = d
        elif i == j:
            out[i,j] = 0.


@cuda.jit
def distance_matrix_XY_gpu(X, Y, out, distance_measure=0):
    i,j = cuda.grid(2)
    nX = X.shape[0]
    nY = Y.shape[0]
    if (i < nX) and (j < nY):
        out[i,j] = calc_distance(X[i], Y[j], distance_measure)
                
        
@cuda.jit
def distance_matrix_condensed_gpu(X, out, distance_measure=0):
    i,j = cuda.grid(2)
    n = X.shape[0]
    if (i < n) and (j > i) and (j < n):
        out[condensed_idx(i,j,n)] = calc_distance(X[i], X[j], distance_measure)
        
        
@cuda.jit
def distance_matrix_XY_part_of_symmetric_gpu(start_row, start_col, X, Y, out, distance_measure=0):
    i,j = cuda.grid(2)
    nX = X.shape[0]
    nY = Y.shape[0]
    global_i = start_row + i
    global_j = start_col + j
    if (i < nX) and (j < nY) and (global_i < global_j):
        out[i,j] = calc_distance(X[i], Y[j], distance_measure)

                
@njit(parallel=True)
def distance_matrix_euclidean2_cpu(X):
    n = X.shape[0]
    out = np.dot(X, X.T)
    for i in prange(n):
        for j in range(i+1,n):
            out[i,j] = out[i,i] - 2.*out[i,j] + out[j,j]
            out[j,i] = out[i,j]
    np.fill_diagonal(out, 0.)
    return out


@njit(parallel=True)
def distance_matrix_euclidean2_XY_cpu(X,Y):
    nX = X.shape[0]
    nY = Y.shape[0]
    out = np.dot(X, Y.T)
    NX = np.sum(X*X, axis=1)
    NY = np.sum(Y*Y, axis=1)
    for i in prange(nX):
        for j in range(nY):
            out[i,j] = NX[i] - 2.*out[i,j] + NY[j]
    return out


@njit(parallel=True)
def distance_matrix_euclidean2_XY_weighted_cpu(X, Y, weightsX, weightsY):
    nX = X.shape[0]
    nY = Y.shape[0]
    n_weightsX = weightsX.shape[0]
    n_weightsY = weightsY.shape[0]
    weighted = (n_weightsX > 0) and (n_weightsY > 0)
    out = np.dot(X, Y.T)
    NX = np.sum(X*X, axis=1)
    NY = np.sum(Y*Y, axis=1)    
    if weighted:
        for i in prange(nX):
            for j in range(nY):
                out[i,j] = (NX[i] - 2. * out[i,j] + NY[j]) * weightsX[i] * weightsY[j]
    else:    
        for i in prange(nX):
            for j in range(nY):
                out[i,j] = NX[i] - 2. * out[i,j] + NY[j]
    return out
   
    
@njit(parallel=True)
def distance_matrix_euclidean2_condensed_cpu(X):
    n = X.shape[0]
    condensed_len = int((n*(n-1))/2)
    out = np.empty(condensed_len, dtype = X.dtype)
    gram = np.dot(X, X.T)
    for i in prange(n):
        for j in range(i+1,n):
            out[condensed_idx(i,j,n)] = gram[i,i] - 2.*gram[i,j] + gram[j,j]
    return out


@njit(parallel=True)
def distance_matrix_cosine_cpu(X):
    n = X.shape[0]
    out = np.dot(X, X.T)
    for i in prange(n):
        for j in range(i+1,n):
            if out[i,i]==0. or out[j,j]==0.:
                out[i,j] = 1.
            else:
                out[i,j] = abs(1.-out[i,j] / (out[i,i] * out[j,j]) ** .5) # Return absolute value to avoid small negative value due to rounding
            out[j,i] = out[i,j]
    np.fill_diagonal(out, 0.)
    return out


@njit(parallel=True)
def distance_matrix_cosine_XY_cpu(X, Y):
    nX = X.shape[0]
    nY = Y.shape[0]
    out = np.dot(X, Y.T)
    NX = np.sum(X*X, axis=1)
    NY = np.sum(Y*Y, axis=1)
    for i in prange(nX):
        for j in range(nY):
            if NX[i]==0. or NY[j]==0.:
                out[i,j] = 1.
            else:
                out[i,j] = abs(1.0-out[i,j] / (NX[i] * NY[j]) ** .5) # Return absolute value to avoid small negative value due to rounding
    return out


@njit(parallel=True)
def distance_matrix_cosine_XY_weighted_cpu(X, Y, weightsX, weightsY):
    nX = X.shape[0]
    nY = Y.shape[0]
    n_weightsX = weightsX.shape[0]
    n_weightsY = weightsY.shape[0]
    weighted = (n_weightsX > 0) and (n_weightsY > 0)    
    out = np.dot(X, Y.T)
    NX = np.sum(X*X, axis=1)
    NY = np.sum(Y*Y, axis=1)
    if weighted:
        for i in prange(nX):
            for j in range(nY):
                if NX[i]==0. or NY[j]==0.:
                    out[i,j] = 1. * weightsX[i] * weightsY[j]
                else:
                    out[i,j] = abs(1.0-out[i,j] / (NX[i] * NY[j]) ** .5) * weightsX[i] * weightsY[j]
    else:        
        for i in prange(nX):
            for j in range(nY):
                if NX[i]==0. or NY[j]==0.:
                    out[i,j] = 1.
                else:
                    out[i,j] = abs(1.0-out[i,j] / (NX[i] * NY[j]) ** .5)
    return out


@njit(parallel=True)
def distance_matrix_cosine_condensed_cpu(X):
    n = X.shape[0]
    condensed_len = int((n*(n-1))/2)
    gram = np.dot(X, X.T)
    out = np.empty(condensed_len, dtype = X.dtype)
    for i in prange(n):
        for j in range(i+1,n):
            if gram[i,i]==0. or gram[j,j]==0.:
                out[condensed_idx(i,j,n)] = 1.
            else:
                out[condensed_idx(i,j,n)] = 1.-gram[i,j] / (gram[i,i] * gram[j,j]) ** .5
    return out

                        
@njit(parallel=True)
def distance_matrix_cpu(X, distance_measure=0):
    n = X.shape[0]
    out = np.empty((n,n), dtype = X.dtype)
    for i in prange(n):
        u = X[i]
        out[i,i] = 0.0
        for j in range(i+1, n):
            d = calc_distance(u, X[j], distance_measure)
            out[i,j] = d
            out[j,i] = d
    return out


@njit(parallel=True)
def distance_matrix_XY_cpu(X, Y, distance_measure=0):
    nX = X.shape[0]
    nY = Y.shape[0]
    out = np.empty((nX, nY), dtype = X.dtype)
    for i in prange(nX):
        u = X[i]
        for j in range(0, nY):
            out[i,j] = calc_distance(u, Y[j], distance_measure)
    return out


@njit(parallel=True)
def distance_matrix_condensed_cpu(X, distance_measure=0):
    n = X.shape[0]
    condensed_len = int((n*(n-1))/2)
    out = np.empty(condensed_len, dtype = X.dtype)
    for i in prange(n):
        u = X[i]
        for j in range(i+1, n):
            out[condensed_idx(i,j,n)] = calc_distance(u, X[j], distance_measure)
    return out


def pairwise_distances_cpu(X, Y = None, distance_measure=0, condensed = True):
    assert ((Y is None) or (X.dtype == Y.dtype)) and (X.dtype == np.float32 or X.dtype == np.float64)
    if distance_measure == 0:
        if Y is None:
            if condensed:
                D = distance_matrix_euclidean2_condensed_cpu(X)
            else:                
                D = distance_matrix_euclidean2_cpu(X)
        else:
            D = distance_matrix_euclidean2_XY_cpu(X,Y)
                         
    else:
        if Y is None:
            if condensed:
                D = distance_matrix_cosine_condensed_cpu(X)
            else:                
                D = distance_matrix_cosine_cpu(X)
        else:
            D = distance_matrix_cosine_XY_cpu(X,Y)
    return D

        

@cuda.jit
def distance_matrix_euclidean2_XY_gpu(X, Y, NX, NY, out):
    i,j = cuda.grid(2)
    nX = X.shape[0]
    nY = Y.shape[0]
    if (i < nX) and (j < nY):
        out[i,j] = NX[i] - 2.*out[i,j] + NY[j]
        
        
# Diagonal must be filled separately by zeros
@cuda.jit
def distance_matrix_euclidean2_gpu(X, out):
    i,j = cuda.grid(2)
    n = X.shape[0]    
    if (i < n) and (j < n) and (j > i):
        d = out[i,i] - 2.*out[i,j] + out[j,j]
        out[i,j] = d
        out[j,i] = d
        
        
@cuda.jit
def distance_matrix_euclidean2_condensed_gpu(X, gram, out):
    i,j = cuda.grid(2)
    n = X.shape[0]
    if (i < n) and (j > i) and (j < n):
        out[condensed_idx(i,j,n)] = gram[i,i] - 2.*gram[i,j] + gram[j,j]
        
        
@cuda.jit
def distance_matrix_cosine_XY_gpu(X, Y, NX, NY, out):
    i,j = cuda.grid(2)
    nX = X.shape[0]
    nY = Y.shape[0]
    if (i < nX) and (j < nY):
        if NX[i]==0. or NY[j]==0.:
            out[i,j] = 1.
        else:
            out[i,j] = 1.-out[i,j] / (NX[i] * NY[j]) ** .5           

           
        
# Diagonal must be filled separately by zeros
@cuda.jit
def distance_matrix_cosine_gpu(X, out):
    i,j = cuda.grid(2)
    n = X.shape[0]
    if (i < n) and (j < n) and (j > i):
        if out[i,i]==0. or out[j,j]==0.:
            out[i,j] = 1.
        else:
            out[i,j] = 1.-out[i,j] / (out[i,i] * out[j,j]) ** .5
                        
            
@cuda.jit
def distance_matrix_cosine_condensed_gpu(X, gram, out):
    i,j = cuda.grid(2)
    n = X.shape[0]
    if (i < n) and (j > i) and (j < n):
        if gram[i,i]==0. or gram[j,j]==0.:
            out[condensed_idx(i,j,n)] = 1.
        else:
            out[condensed_idx(i,j,n)] = 1.-gram[i,j] / (gram[i,i] * gram[j,j]) ** .5
            
            
def pairwise_distances_gpu(X, Y = None, distance_measure=0, condensed = True, gpu_device_id = 0, threads_per_block = (4, 16)):
    assert (len(X.shape) == 2) and (X.shape[0] > 0)
    available_gpu_ids = set([gpu.id for gpu in nb.cuda.gpus.lst])
    assert (gpu_device_id > -1) and (gpu_device_id in available_gpu_ids)
    assert ((Y is None) or (X.dtype == Y.dtype)) and (X.dtype == np.float32 or X.dtype == np.float64)
    
    gpu = nb.cuda.select_device(gpu_device_id)
    cp.cuda.Device(gpu_device_id).use()
    
    nX = X.shape[0]
    X_gpu = cp.asarray(X)

    if Y is None:
        nY = 0
        grid_dim = (int(nX/threads_per_block[0] + 1), int(nX/threads_per_block[1] + 1))
        
        if condensed:
            condensed_len = condensed_size(n_rowsX)
            gram_gpu = X_gpu.dot(X_gpu.T)
            out_gpu = cp.empty(condensed_len, dtype = X_gpu.dtype)
            if distance_measure == 0:
                distance_matrix_euclidean2_condensed_gpu[grid_dim, threads_per_block](X_gpu, gram_gpu, out_gpu)
            else:
                distance_matrix_cosine_condensed_gpu[grid_dim, threads_per_block](X_gpu, gram_gpu, out_gpu)
        else:
            out_gpu = X_gpu.dot(X_gpu.T)
            if distance_measure == 0:
                distance_matrix_euclidean2_gpu[grid_dim, threads_per_block](X_gpu, out_gpu)
            else:
                distance_matrix_cosine_gpu[grid_dim, threads_per_block](X_gpu, out_gpu)
            cp.fill_diagonal(out_gpu, 0.)
                
    else:
        assert (len(Y.shape) == 2) and (Y.shape[0] > 0) and (X.shape[1]==Y.shape[1])
        nY = Y.shape[0]
        Y_gpu = cp.asarray(Y)
        grid_dim = (int(nX/threads_per_block[0] + 1), int(nY/threads_per_block[1] + 1))
        
        out_gpu = cp.dot(X_gpu, Y_gpu.T)
        NX_gpu = cp.sum(X_gpu*X_gpu, axis=1)
        NY_gpu = cp.sum(Y_gpu*Y_gpu, axis=1)
                
        if distance_measure == 0:
            distance_matrix_euclidean2_XY_gpu[grid_dim, threads_per_block](X_gpu, Y_gpu, NX_gpu, NY_gpu, out_gpu)
        else:
            distance_matrix_cosine_XY_gpu[grid_dim, threads_per_block](X_gpu, Y_gpu, NX_gpu, NY_gpu, out_gpu)
            
    out = cp.asnumpy(out_gpu)

    return out
                        

# gpu_device_id - ID of GPU divice that will be used for perform calculations
# if gpu_device_id = -1 then the CPUs will be used for calculations
def distance_matrix(X, Y = None, distance_measure=0, condensed = True, gpu_device_id = -1, threads_per_block = (4, 16)):   
    if gpu_device_id > -1:
        pairwise_distances_gpu(X, Y, distance_measure, condensed, gpu_device_id, threads_per_block)                
    else:
        out = pairwise_distances_cpu(X, Y, distance_measure, condensed)
    return out


# # https://stackoverflow.com/questions/58002793/how-can-i-use-multiple-gpus-in-cupy
# # https://github.com/numba/numba/blob/master/numba/cuda/tests/cudapy/test_multigpu.py
# # usage
# def pairwise_distances_multigpu(X, Y = None, distance_measure=0, devices = [], memory_usage = 0.95, threads_per_block = (4, 16)):
#     assert ((Y is None) or (X.dtype == Y.dtype)) and (X.dtype == np.float32 or X.dtype == np.float64)
#     assert memory_usage > 0. and memory_usage <= 1.
    
#     nX = X.shape[0]
#     n_devices = len(devices)
    
#     available_devices = [gpu.id for gpu in nb.cuda.gpus.lst]
    
#     if n_devices == 0:
#         used_devices = available_devices
#     else:
#         used_devices = list(set(devices).intersection(set(available_devices)))
    
#     n_used_devices = len(used_devices)
    
#     capacities = np.empty(n_used_devices)
    
#     for i in range(n_used_devices):
#         capacities[i] = nb.cuda.current_context(used_devices[i]).get_memory_info().free * memory_usage
        
#     full_capacity = np.sum(capacities)
#     fractions = capacities / full_capacity
        
#     n_elements = condensed_size(nX)
#     if X.dtype == np.float32:
#         n_bytes = n_elements * 4
#     else:
#         n_bytes = n_elements * 8
        
#     n_portions = n_bytes / full_capacity


######################################
#Multi-GPU distance matrix calculation
######################################
# Split the dataset into portions and calculate the distance matrix for each portion on multiple GPUs in parallel.
# X: array of pairwise distances between samples, or a feature array;
# Y: an optional second feature array;
# D: a distance matrix D such that D_{i, j} is the distance between the ith and jth vectors of the given matrix X, if Y is None. If Y is not None, then D_{i, j} is the distance between the ith array from X and the jth array from Y.
# sizeX: portion size for X;
# sizeY: portion size for Y;
# If condensed = True, then condenced representation for matrix D will be used;
# distance_measure: 0 - Squared Euclidean distance; 1 - Euclidean distance; 2 - cosine distance;
# gpu_device_ids: list of GPU ids that will be used for calculations
def distance_matrix_multi_gpu(X, Y = None, sizeX = None, sizeY = None, condensed = True, distance_measure=0, gpu_device_ids = [], show_progress = False, threads_per_block = (4, 16)):
            
    @njit(parallel = True)
    def aggregation_1D_1D(out, size, sub_mat, row, col):
        sub_mat_condensed_len = sub_mat.shape[0]
        sub_mat_size = matrix_size(sub_mat)
        for i in prange(sub_mat_condensed_len):
            x,y = regular_idx(i, sub_mat_size)
            x += row
            y += col
            if x < y:
                out[condensed_idx(x,y,size)] = sub_mat[i]
                
    @njit(parallel = True)
    def aggregation_1D_2D(out, size, sub_mat, row, col):
        n_rows, n_cols = sub_mat.shape
        for i in prange(n_rows):
            for j in range(n_cols):
                x = row + i
                y = col + j
                if x < y:
                    out[condensed_idx(x,y,size)] = sub_mat[i,j]

    @njit(parallel = True)
    def aggregation_2D_1D(out, sub_mat, row, col):
        sub_mat_condensed_len = sub_mat.shape[0]
        sub_mat_size = matrix_size(sub_mat)
        for i in prange(sub_mat_condensed_len):
            x,y = regular_idx(i, sub_mat_size)
            x += row
            y += col
            out[x,y] = sub_mat[i]
            out[y,x] = sub_mat[i]
            
    @njit(parallel = True)
    def aggregation_2D_2D_symmetric(out, sub_mat, row, col):
        n_rows, n_cols = sub_mat.shape
        for i in prange(n_rows):
            for j in range(n_cols):
                x = row + i
                y = col + j
                if x < y:
                    out[x,y] = sub_mat[i,j]
                    out[y,x] = sub_mat[i,j]
                    
    @njit(parallel = True)
    def aggregation_2D_2D_asymmetric(out, sub_mat, row, col):
        n_rows, n_cols = sub_mat.shape
        for i in prange(n_rows):
            for j in range(n_cols):
                x = row + i
                y = col + j
                out[x,y] = sub_mat[i,j]
                            
                                
    def calc_submatrix(X, Y, i, j, row1, row2, col1, col2, out, threads_per_block, distance_measure):
        NX = X.shape[0]
        nX = row2-row1
        nY = col2-col1
        symmetric = Y is None
        is_condensed_out = out.ndim == 1
       
        stream = cuda.stream()
        grid_dim = (int(nX/threads_per_block[0] + 1), int(nY/threads_per_block[1] + 1))
        X_cu = cuda.to_device(X[row1:row2], stream=stream)

        if symmetric and (row1 < col2-1):

            if (i == j) and (nX == nY):
                sub_mat_cu = cuda.device_array(shape=int((nX*(nX-1))/2), dtype = X_cu.dtype, stream=stream)
                distance_matrix_condensed_gpu[grid_dim, threads_per_block, stream](X_cu, sub_mat_cu, distance_measure)
                sub_mat = sub_mat_cu.copy_to_host(stream=stream)
                if is_condensed_out:
                    aggregation_1D_1D(out, NX, sub_mat, row1, col1)
                else:
                    aggregation_2D_1D(out, sub_mat, row1, col1)

            else:
                Y_cu = cuda.to_device(X[col1:col2], stream=stream)                    
                sub_mat_cu = cuda.device_array(shape=(nX,nY), dtype = X_cu.dtype, stream=stream)
                distance_matrix_XY_part_of_symmetric_gpu[grid_dim, threads_per_block, stream](row1, col1,  X_cu, Y_cu, sub_mat_cu, distance_measure)
                sub_mat = sub_mat_cu.copy_to_host(stream=stream)
                if is_condensed_out:
                    aggregation_1D_2D(out, NX, sub_mat, row1, col1)
                else:
                    aggregation_2D_2D_symmetric(out, sub_mat, row1, col1)


        elif (not symmetric):
          
            Y_cu = cuda.to_device(Y[col1:col2], stream=stream)
            sub_mat_cu = cuda.device_array(shape=(nX,nY), dtype = Y_cu.dtype, stream=stream)
            distance_matrix_XY_gpu[grid_dim, threads_per_block, stream](X_cu, Y_cu, sub_mat_cu, distance_measure)

            sub_mat = sub_mat_cu.copy_to_host(stream=stream)
            aggregation_2D_2D_asymmetric(out, sub_mat, row1, col1)
        
        
    def calc_portion(portion, bounds, X, Y, out, device_id, threads_per_block, distance_measure):
        if device_id > -1:
            gpu = nb.cuda.select_device(device_id)
        for B in [bounds[i] for i in portion]:
            calc_submatrix(X, Y, B[0], B[1], B[2], B[3], B[4], B[5], out, threads_per_block, distance_measure)
            if show_progress:
                print(device_id, B)
                
                
    available_gpu_device_ids = [gpu.id for gpu in nb.cuda.gpus.lst]
    
    symmetric = Y is None
    assert (len(X.shape) == 2) and (X.shape[0] > 0)
    assert symmetric or ((len(Y.shape) == 2) and (Y.shape[0] > 0) and (Y.shape[1]==X.shape[1]))
    
    NX = X.shape[0]
    if (sizeX is None) or (sizeX < 1) or (sizeX > NX):
        sizeX = NX
    else:
        sizeX = int(sizeX)
    n_partsX = math.ceil(NX / sizeX)
    
    if symmetric:
        NY = NX
        sizeY = sizeX
        n_partsY = n_partsX
    else:
        NY = Y.shape[0]
        if (sizeY is None) or (sizeY < 1) or (sizeY > NY):
            sizeY = NY
        else:        
            sizeY = int(sizeY)
        n_partsY = math.ceil(NY / sizeY)
    
    if condensed and symmetric:
        out = np.empty(shape = int((NX*(NX-1))/2), dtype = X.dtype)
    else:
        out = np.empty((NX,NY), dtype = X.dtype)

    
    bounds = []
    for i in range(n_partsX):
        row1 = i*sizeX
        row2 = min(row1 + sizeX, NX)       
                
        for j in range(n_partsY):            
            col1 = j*sizeY
            col2 = min(col1 + sizeY, NY)            
            
            bounds.append((i, j, row1, row2, col1, col2))
     
    n_bounds = len(bounds)
    if n_bounds > 0:
        used_gpu_device_ids = list(set(gpu_device_ids).intersection(set(available_gpu_device_ids)))
        n_gpu = len(used_gpu_device_ids)
        if (n_gpu > 0) and (n_bounds > 1):
            
            sequence = np.random.permutation(n_bounds)
            
            if n_bounds < n_gpu:
                n_portions = n_bounds
                portion_size = 1
            else:
                n_portions = n_gpu
                portion_size = n_bounds // n_gpu
                
            portions = []
            for i in range(n_portions):
                a = i * portion_size
                if i < n_portions-1:
                    b = a + portion_size
                else:
                    b = n_bounds
                portions.append(sequence[a:b])
            
            threads = [threading.Thread(target=calc_portion, args=(portions[i], bounds, X, Y, out, used_gpu_device_ids[i], threads_per_block, distance_measure)) for i in range(n_portions)]
            
            for th in threads:
                th.start()

            for th in threads:
                th.join()            
            
        else:
            for B in bounds:
                calc_submatrix(X, Y, B[0], B[1], B[2], B[3], B[4], B[5], out, threads_per_block, distance_measure)                   
                
    return out
                    
                    
                    
@njit
def search_sorted(X, val):
    n = X.shape[0]
    ind = -1
    if n > 0:
        for i in range(0,n):
            if val <= X[i]:
                ind = i
                break
        if ind == -1:
            ind = n
    return ind


@njit
def cum_sum(X, cumsum):
    summ = 0.0
    for i in range(X.shape[0]):
        if math.isnan(X[i]):
            cumsum[i] = X[i]
        else:
            summ += X[i]
            cumsum[i] = summ

        
@njit
def random_choice(cumsum):
    potential = cumsum[cumsum.shape[0]-1]
    rand_val = np.random.random_sample() * potential
    ind = search_sorted(cumsum, rand_val)
    max_ind = cumsum.shape[0]-1
    if ind > max_ind:
        ind = max_ind    
    return ind


@njit
def cum_search(X, vals, out):
    n = X.shape[0]
    n_vals = vals.shape[0]
    assert n>0 and n_vals == out.shape[0] and n_vals>0
    cum_sum = 0.0
    ind_vals = 0
    sorted_inds = np.argsort(vals)
    for i in range(n):
        if not math.isnan(X[i]):
            cum_sum += X[i]
            while vals[sorted_inds[ind_vals]] <= cum_sum:
                out[sorted_inds[ind_vals]] = i
                ind_vals += 1
                if ind_vals == n_vals:
                    return
    out[sorted_inds[ind_vals: n_vals]] = n-1


@njit(parallel = True)
def k_means_pp_naive(samples, sample_weights, n_clusters, distance_measure=0):
    n_samples, n_features = samples.shape
    n_sample_weights, = sample_weights.shape
    assert ((n_samples == n_sample_weights) or (n_sample_weights == 0))
    centroid_inds = np.full(n_clusters, -1)    
    if (n_samples > 0) and (n_features > 0):
        if n_sample_weights > 0:
            weights = np.copy(sample_weights)
        else:
            weights = np.full(n_samples, 1.0)
        
        cumsum = np.empty(n_samples)
        cum_sum(weights, cumsum)
        new_centroid = random_choice(cumsum)
        
        n_centroids = 0
        centroid_inds[n_centroids] = new_centroid
        n_centroids += 1        
        while n_centroids < n_clusters:
            for i in prange(n_samples):
                weights[i] = 0.0
            for i in prange(n_samples):
                min_dist = np.inf
                for j in range(n_centroids):
                    dist = calc_distance(samples[i], samples[centroid_inds[j]], distance_measure)
                    dist *= sample_weights[i]*sample_weights[centroid_inds[j]]
                    if dist < min_dist:
                        min_dist = dist
                if min_dist < np.inf:
                    weights[i] = min_dist
            
            cum_sum(weights, cumsum)
            new_centroid = random_choice(cumsum)
                        
            centroid_inds[n_centroids] = new_centroid
            n_centroids += 1    
    return centroid_inds



@njit(parallel=True)
def additional_centers_naive(samples, sample_weights, centroids, n_additional_centers=1, distance_measure=0):
    n_samples, n_features = samples.shape
    n_sample_weights, = sample_weights.shape
    n_centers = centroids.shape[0]
    
    assert ((n_samples == n_sample_weights) or (n_sample_weights == 0))
    
    center_inds = np.full(n_additional_centers, -1)
    
    nondegenerate_mask = np.sum(np.isnan(centroids), axis = 1) == 0
    
    n_nondegenerate_clusters = np.sum(nondegenerate_mask)
    
    center_inds = np.full(n_additional_centers, -1)
    if (n_samples > 0) and (n_features > 0) and (n_additional_centers > 0):
                
        cumsum = np.empty(n_samples)
        
        n_added_centers = 0
                
        if n_nondegenerate_clusters == 0:           
            if n_sample_weights > 0:
                cum_sum(sample_weights, cumsum)
                center_inds[0] = random_choice(cumsum)
            else:
                center_inds[0] = np.random.randint(n_samples)                
            n_added_centers += 1       
                
        nondegenerate_centroids = centroids[nondegenerate_mask]
        weights = np.empty(n_samples)
        for c in range(n_added_centers, n_additional_centers):
            
            for i in prange(n_samples):
                weights[i] = 0.0
                
            for i in prange(n_samples):
                min_dist = np.inf
                
                for j in range(n_nondegenerate_clusters):
                    dist = calc_distance(samples[i], nondegenerate_centroids[j], distance_measure)
                   
                    if dist < min_dist:
                        min_dist = dist
                        
                for j in range(n_added_centers):
                    dist = calc_distance(samples[i], samples[center_inds[j]], distance_measure)
                    
                    if dist < min_dist:
                        min_dist = dist                                               
                        
                if min_dist < np.inf:
                    weights[i] = min_dist * sample_weights[i]
            
            cum_sum(weights, cumsum)
            new_centroid = random_choice(cumsum)
            center_inds[c] = new_centroid
            
            n_added_centers += 1
            
    return center_inds
  
    
# k-means++ : algorithm for choosing the initial cluster centers (or "seeds") for the k-means clustering algorithm
# samples должны быть хорошо перемешаны ??????? !!!!!!!!!!!!!!!!!!
@njit(parallel=True)
def k_means_pp(samples, sample_weights, n_centers=3, n_candidates=3, distance_measure=0):
    n_samples, n_features = samples.shape
    n_sample_weights, = sample_weights.shape
    
    assert ((n_samples == n_sample_weights) or (n_sample_weights == 0))
    center_inds = np.full(n_centers, -1)
    if (n_samples > 0) and (n_features > 0) and (n_centers > 0):
        
        if (n_candidates <= 0) or (n_candidates is None):
            n_candidates = 2 + int(np.log(n_centers))               
                
        if n_sample_weights > 0:
            cumsum = np.empty(n_samples)
            cum_sum(sample_weights, cumsum)
            center_inds[0] = random_choice(cumsum)
        else:
            center_inds[0] = np.random.randint(n_samples)
                
        #dist_mat = np.empty((1,n_samples))
        indices = np.full(1, center_inds[0])
        if distance_measure == 1:
            dist_mat = distance_matrix_cosine_XY_weighted_cpu(samples[indices], samples, sample_weights[indices], sample_weights)
        else:
            dist_mat = distance_matrix_euclidean2_XY_weighted_cpu(samples[indices], samples, sample_weights[indices], sample_weights)
        
        closest_dist_sq = dist_mat[0]
        
        current_pot = 0.0
        for i in prange(n_samples):
            current_pot += closest_dist_sq[i]
                                      
        candidate_ids = np.full(n_candidates, -1)
                
        #distance_to_candidates = np.empty((n_candidates,n_samples))
        candidates_pot = np.empty(n_candidates)
                
        for c in range(1,n_centers):
            rand_vals = np.random.random_sample(n_candidates) * current_pot
            
            cum_search(closest_dist_sq, rand_vals, candidate_ids)
                
            
            if distance_measure == 1:
                distance_to_candidates = distance_matrix_cosine_XY_weighted_cpu(samples[candidate_ids], samples, sample_weights[candidate_ids], sample_weights)
            else:
                distance_to_candidates = distance_matrix_euclidean2_XY_weighted_cpu(samples[candidate_ids], samples, sample_weights[candidate_ids], sample_weights)
            
            for i in prange(n_candidates):
                candidates_pot[i] = 0.0
                for j in range(n_samples):
                    distance_to_candidates[i,j] = min(distance_to_candidates[i,j],closest_dist_sq[j])
                    candidates_pot[i] += distance_to_candidates[i,j]
            
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            
            for i in prange(n_samples):
                closest_dist_sq[i] = distance_to_candidates[best_candidate][i]
            
            center_inds[c] = candidate_ids[best_candidate]

    return center_inds


# choosing the new n additional centers for existing ones using the k-means++ logic
@njit(parallel=True)
def additional_centers(samples, sample_weights, centroids, n_additional_centers=3, n_candidates=3, distance_measure=0):
    n_samples, n_features = samples.shape
    n_sample_weights, = sample_weights.shape
    n_centers = centroids.shape[0]
    
    assert ((n_samples == n_sample_weights) or (n_sample_weights == 0))
    
    center_inds = np.full(n_additional_centers, -1)
    
    nondegenerate_mask = np.sum(np.isnan(centroids), axis = 1) == 0
    
    n_nondegenerate_clusters = np.sum(nondegenerate_mask)
    
    center_inds = np.full(n_additional_centers, -1)
    if (n_samples > 0) and (n_features > 0) and (n_additional_centers > 0):
        
        if (n_candidates <= 0) or (n_candidates is None):
            n_candidates = 2 + int(np.log(n_nondegenerate_clusters+n_additional_centers))
                                       
        if n_nondegenerate_clusters > 0:
            closest_dist_sq = np.full(n_samples, np.inf)
            #distance_to_centroids = np.empty((n_nondegenerate_clusters, n_samples))
            
            centroid_weights = np.ones(n_nondegenerate_clusters)
            
            if distance_measure == 1:
                distance_to_centroids = distance_matrix_cosine_XY_weighted_cpu(centroids[nondegenerate_mask], samples, centroid_weights, sample_weights)
            else:
                distance_to_centroids = distance_matrix_euclidean2_XY_weighted_cpu(centroids[nondegenerate_mask], samples, centroid_weights, sample_weights)
                    
            current_pot = 0.0
            for i in prange(n_samples):
                for j in range(n_nondegenerate_clusters):
                    closest_dist_sq[i] = min(distance_to_centroids[j,i],closest_dist_sq[i])
                current_pot += closest_dist_sq[i]
                
            n_added_centers = 0
        
        else:
            
            if n_sample_weights > 0:
                cumsum = np.empty(n_samples)
                cum_sum(sample_weights, cumsum)
                center_inds[0] = random_choice(cumsum)
            else:
                center_inds[0] = np.random.randint(n_samples)

            #dist_mat = np.empty((1,n_samples))
            indices = np.full(1, center_inds[0])
            if distance_measure == 1:
                dist_mat = distance_matrix_cosine_XY_weighted_cpu(samples[indices], samples, sample_weights[indices], sample_weights)
            else:
                dist_mat = distance_matrix_euclidean2_XY_weighted_cpu(samples[indices], samples, sample_weights[indices], sample_weights)

            closest_dist_sq = dist_mat[0]
            
            current_pot = 0.0
            for i in prange(n_samples):
                current_pot += closest_dist_sq[i]
                
            n_added_centers = 1        
        
        candidate_ids = np.full(n_candidates, -1)
                
        #distance_to_candidates = np.empty((n_candidates,n_samples))
        candidates_pot = np.empty(n_candidates)
                
        for c in range(n_added_centers, n_additional_centers):
            
            rand_vals = np.random.random_sample(n_candidates) * current_pot
            
            cum_search(closest_dist_sq, rand_vals, candidate_ids)
                
            if distance_measure == 1:
                distance_to_candidates = distance_matrix_cosine_XY_weighted_cpu(samples[candidate_ids], samples, sample_weights[candidate_ids], sample_weights)
            else:
                distance_to_candidates = distance_matrix_euclidean2_XY_weighted_cpu(samples[candidate_ids], samples, sample_weights[candidate_ids], sample_weights)
            
            for i in prange(n_candidates):
                candidates_pot[i] = 0.0
                for j in range(n_samples):
                    distance_to_candidates[i,j] = min(distance_to_candidates[i,j],closest_dist_sq[j])
                    candidates_pot[i] += distance_to_candidates[i,j]
            
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            
            for i in prange(n_samples):
                closest_dist_sq[i] = distance_to_candidates[best_candidate][i]
            
            center_inds[c] = candidate_ids[best_candidate]
                        
    return center_inds


@njit
def check_shapes(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives):
    assert samples.ndim == 2
    n_samples, n_features = samples.shape
    assert (sample_weights.shape == (n_samples,)) or (sample_weights.shape == (0,))
    assert sample_membership.shape == (n_samples,)
    assert (sample_objectives.shape == (n_samples,)) or (sample_objectives.shape[0] == 0)
    assert (centroids.ndim == 2) and (centroids.shape[1] == n_features)
    n_clusters = centroids.shape[0]
    assert centroid_sums.shape == (n_clusters, n_features)
    assert centroid_counts.shape == (n_clusters, )
    assert centroid_objectives.shape == (n_clusters,)

               
@njit
def empty_state(n_samples, n_features, n_clusters):
    sample_weights = np.ones(n_samples)
    sample_membership = np.full(n_samples, -1)
    sample_objectives = np.full(n_samples, np.nan)
    centroids = np.full((n_clusters, n_features), np.nan)
    centroid_sums = np.full((n_clusters, n_features), np.nan)
    centroid_counts = np.full(n_clusters, 0.0)
    centroid_objectives = np.full(n_clusters, np.nan)
    return sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives


@njit
def sub_sections(n_samples, n_sections):
    n_samples = int(abs(n_samples)) # Распространить этот подход на другие функции
    n_sections = int(abs(n_sections))
    samples_per_section, n_extras = divmod(n_samples, n_sections)    
    if samples_per_section == 0:
        n_sections = n_extras        
    points = np.full(n_sections, samples_per_section)
    for i in range(n_extras):
        points[i] += 1        
    cumsum = 0
    for i in range(n_sections):
        cumsum += points[i]
        points[i] = cumsum        
    sections = np.empty((n_sections,2), dtype = points.dtype)    
    start_ind = 0
    for i in range(n_sections):
        sections[i,0] = start_ind
        sections[i,1] = points[i]
        start_ind = points[i]                
    return sections
                
                
@njit
def random_membership(n_samples, n_clusters, sample_membership):
    sample_membership[:] = np.random.randint(0, n_clusters, n_samples)
    
    
@njit(parallel = True)
def initialize_by_membership(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives):
    n_samples, n_features = samples.shape
    n_clusters = centroids.shape[0]
    n_sample_weights = sample_weights.shape[0]
    
    centroids.fill(np.nan)
    centroid_sums.fill(0.0)
    centroid_counts.fill(0.0)
    centroid_objectives.fill(0.0)   
    
    thread_ranges = sub_sections(n_samples, nb.config.NUMBA_NUM_THREADS)
    n_threads = thread_ranges.shape[0]
    thread_centroid_sums = np.zeros((n_threads,n_clusters,n_features))
    thread_centroid_counts = np.zeros((n_threads,n_clusters))
    
    if n_sample_weights > 0:
        for i in prange(n_threads):
            for j in range(thread_ranges[i,0],thread_ranges[i,1]):
                centroid_ind = sample_membership[j]
                for k in range(n_features):
                    thread_centroid_sums[i,centroid_ind,k] += sample_weights[j] * samples[j,k]
                thread_centroid_counts[i,centroid_ind] += sample_weights[j]
    else:
        for i in prange(n_threads):
            for j in range(thread_ranges[i,0],thread_ranges[i,1]):
                centroid_ind = sample_membership[j]
                for k in range(n_features):
                    thread_centroid_sums[i,centroid_ind,k] += samples[j,k]
                thread_centroid_counts[i,centroid_ind] += 1.0
            
    for i in range(n_threads):
        for j in range(n_clusters):
            centroid_counts[j] += thread_centroid_counts[i,j]
            for k in range(n_features):
                centroid_sums[j,k] += thread_centroid_sums[i,j,k]

    for i in range(n_clusters):
        if centroid_counts[i] > 0.0:
            for j in range(n_features):
                centroids[i,j] = centroid_sums[i,j] / centroid_counts[i]
            
    objective = 0.0
    for i in range(n_samples):
        centroid_ind = sample_membership[i]

        dist = 0.0
        for j in range(n_features):
            dist += (centroids[centroid_ind,j]-samples[i,j])**2

        if n_sample_weights > 0:
            sample_objectives[i] = sample_weights[i]*dist
        else:
            sample_objectives[i] = dist
        
        centroid_objectives[centroid_ind] += dist
        objective += dist
            
    return objective


@njit
def cluster_objective_change(sample, sample_weight, centroid, centroid_count):
    n_features = sample.shape[0]
    dist = 0.0
    for i in range(n_features):
        dist += (centroid[i] - sample[i])**2
    return centroid_count/(centroid_count+sample_weight)*dist


@njit
def reallocation_effect(sample, sample_weight, origin_centroid, origin_centroid_counts, destination_centroid, destination_centroid_counts):    
    origin_objective_change = cluster_objective_change(sample,
                                                       -sample_weight,
                                                       origin_centroid,
                                                       origin_centroid_counts)
    
    destination_objective_change = cluster_objective_change(sample,
                                                            sample_weight,
                                                            destination_centroid,
                                                            destination_centroid_counts)

    objective_change = destination_objective_change - origin_objective_change
    return objective_change


# Hartigan–Wong method
# first-improvement strategy
# https://en.wikipedia.org/wiki/K-means_clustering#Hartigan%E2%80%93Wong_method
@njit
def h_means_first(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, max_iters = 300, tol=0.0001):
    n_samples, n_features = samples.shape
    n_clusters = centroids.shape[0]
    n_sample_weights = sample_weights.shape[0]
    
    sample_objectives = np.full(0, np.nan)
    check_shapes(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives)
    
    improved = True
    n_iters = 0
    tolerance = np.inf
    
    while improved and (objective > 0.0) and (n_iters < max_iters) and (tolerance > tol):
        improved = False
        previous_objective = objective
    
        for i in range(n_samples):
            sample = samples[i]
            if n_sample_weights > 0:
                sample_weight = sample_weights[i]
            else:
                sample_weight = 1.0
            weighted_sample = sample*sample_weight

            cluster_i = sample_membership[i]
            count_i = centroid_counts[cluster_i]
            
            if count_i-sample_weight != 0.0:
                objective_change_i = cluster_objective_change(sample, -sample_weight, centroids[cluster_i], count_i)
            else:
                objective_change_i = centroid_objectives[cluster_i]

            best_cluster_j = -1
            best_objective_change = np.inf
            best_objective_change_j = np.nan

            for cluster_j in range(n_clusters):
                if cluster_j != cluster_i:
                    count_j = centroid_counts[cluster_j]

                    if count_j+sample_weight != 0.0:

                        objective_change_j = cluster_objective_change(sample, sample_weight, centroids[cluster_j], count_j)

                        objective_change = objective_change_j - objective_change_i

                        if objective_change < best_objective_change:                                
                            best_cluster_j = cluster_j
                            best_objective_change_j = objective_change_j
                            best_objective_change = objective_change

            if (best_cluster_j > -1) and (best_objective_change < 0.0):
                               
                centroid_sums[cluster_i] -= weighted_sample
                centroid_counts[cluster_i] -= sample_weight
                if centroid_counts[cluster_i] != 0.0:
                    centroids[cluster_i] = centroid_sums[cluster_i]/centroid_counts[cluster_i]
                else:
                    centroids[cluster_i].fill(np.nan)
                centroid_objectives[cluster_i] -= objective_change_i
                
                sample_membership[i] = best_cluster_j
                
                centroid_sums[best_cluster_j] += weighted_sample
                centroid_counts[best_cluster_j] += sample_weight
                if centroid_counts[best_cluster_j] != 0.0:
                    centroids[best_cluster_j] = centroid_sums[best_cluster_j]/centroid_counts[best_cluster_j]
                else:
                    centroids[best_cluster_j].fill(np.nan)                    
                centroid_objectives[best_cluster_j] += best_objective_change_j

                objective += best_objective_change

                improved = True
        
        n_iters += 1
        tolerance = 1 - objective/previous_objective
    return objective, n_iters
   
    
# Hartigan–Wong method
# best-improvement strategy
@njit(parallel = True)
def h_means_best(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, max_iters = 3000, tol=0.0001):
    n_samples, n_features = samples.shape
    n_clusters = centroids.shape[0]
    n_sample_weights = sample_weights.shape[0]
    
    sample_objectives = np.full(0,np.nan)
    check_shapes(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives)
    
    improved = True
    n_iters = 0
    tolerance = np.inf
    
    objective_changes = np.full(n_samples, 0.0)
    new_cluster_inds = np.full(n_samples, -1)    
    
    while improved and (objective > 0.0) and (n_iters < max_iters) and (tolerance > tol):
              
        improved = False
        previous_objective = objective
        
        objective_changes.fill(0.0)
        new_cluster_inds.fill(-1)
    
        for i in prange(n_samples):
            p_sample = samples[i]
            if n_sample_weights > 0:
                p_sample_weight = sample_weights[i]
            else:
                p_sample_weight = 1.0

            p_cluster_i = sample_membership[i]
            p_count_i = centroid_counts[p_cluster_i]            
            
            if p_count_i-p_sample_weight != 0.0:
                p_objective_change_i = cluster_objective_change(p_sample, -p_sample_weight, centroids[p_cluster_i], p_count_i)
            else:
                p_objective_change_i = centroid_objectives[p_cluster_i]

            p_best_cluster_j = -1
            p_best_objective_change = np.inf
            p_best_objective_change_j = np.nan

            for p_cluster_j in range(n_clusters):
                if p_cluster_j != p_cluster_i:
                    p_count_j = centroid_counts[p_cluster_j]

                    if p_count_j+p_sample_weight != 0.0:

                        p_objective_change_j = cluster_objective_change(p_sample, p_sample_weight, centroids[p_cluster_j], p_count_j)

                        p_objective_change = p_objective_change_j - p_objective_change_i

                        if p_objective_change < p_best_objective_change:                                
                            p_best_cluster_j = p_cluster_j
                            p_best_objective_change_j = p_objective_change_j
                            p_best_objective_change = p_objective_change
                            
            if (p_best_cluster_j > -1) and (p_best_objective_change < 0.0):
                objective_changes[i] = p_best_objective_change
                new_cluster_inds[i] = p_best_cluster_j
        

        best_sample_ind = -1
        best_objective_change = np.inf
        for i in range(n_samples):
            if objective_changes[i] < best_objective_change:
                best_sample_ind = i
                best_objective_change = objective_changes[i]
                            

        if (best_sample_ind > -1) and (new_cluster_inds[best_sample_ind] > -1) and (best_objective_change < 0.0):
                                    
            sample = samples[best_sample_ind]
            if n_sample_weights > 0:
                sample_weight = sample_weights[best_sample_ind]
            else:
                sample_weight = 1.0
            weighted_sample = sample_weight * samples[best_sample_ind]
            cluster_i = sample_membership[best_sample_ind]
            cluster_j = new_cluster_inds[best_sample_ind]
            
            if centroid_counts[cluster_j]+sample_weight != 0.0:
            
                if centroid_counts[cluster_i]-sample_weight != 0.0:
                    objective_change_i = cluster_objective_change(sample, -sample_weight, centroids[cluster_i], centroid_counts[cluster_i])
                else:
                    objective_change_i = centroid_objectives[cluster_i]
                centroid_objectives[cluster_i] -= objective_change_i
                
                objective_change_j = cluster_objective_change(sample, sample_weight, centroids[cluster_j], centroid_counts[cluster_j])
                centroid_objectives[cluster_j] += objective_change_j


                centroid_sums[cluster_i] -= weighted_sample
                centroid_counts[cluster_i] -= sample_weight
                if centroid_counts[cluster_i] != 0.0:
                    centroids[cluster_i] = centroid_sums[cluster_i]/centroid_counts[cluster_i]
                else:
                    centroids[cluster_i].fill(np.nan)                               


                sample_membership[best_sample_ind] = cluster_j


                centroid_sums[cluster_j] += weighted_sample
                centroid_counts[cluster_j] += sample_weight
                if centroid_counts[cluster_j] != 0.0:
                    centroids[cluster_j] = centroid_sums[cluster_j]/centroid_counts[cluster_j]
                else:
                    centroids[cluster_j].fill(np.nan)
                

                objective += best_objective_change

                improved = True
           
        n_iters += 1
        tolerance = 1 - objective/previous_objective

        
    return objective, n_iters



def _assignment(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_objectives):
    n_samples, n_features = samples.shape
    n_centroids = centroids.shape[0]
    n_sample_weights = sample_weights.shape[0]
    n_sample_membership = sample_membership.shape[0]
    n_sample_objectives = sample_objectives.shape[0]
    n_centroid_objectives = centroid_objectives.shape[0]

    if n_centroid_objectives > 0:
        centroid_objectives.fill(np.nan)
        
    objective = 0.0
    n_changed_membership = 0
    for i in prange(n_samples):
        min_dist2 = np.inf
        min_ind = -1
        for j in range(n_centroids):
            if not np.isnan(centroids[j,0]):
                dist2 = 0.0
                for h in range(n_features):
                    dist2 += (centroids[j,h] - samples[i,h])**2
                if dist2 < min_dist2:
                    min_dist2 = dist2
                    min_ind = j
        if min_ind == -1: min_dist2 = np.nan
                            
        if (n_sample_membership > 0) and (sample_membership[i] != min_ind):
            n_changed_membership += 1
            sample_membership[i] = min_ind
            
        if n_sample_weights > 0:
            sample_objective = sample_weights[i]*min_dist2
        else:
            sample_objective = min_dist2
            
        if n_sample_objectives > 0:
            sample_objectives[i] = sample_objective
                                        
        if (n_centroid_objectives > 0) and (min_ind > -1):
            if np.isnan(centroid_objectives[min_ind]):
                centroid_objectives[min_ind] = sample_objective
            else:
                centroid_objectives[min_ind] += sample_objective
        
        objective += sample_objective
                
    return objective, n_changed_membership


assignment = njit(parallel=False)(_assignment)
assignment_parallel = njit(parallel=True)(_assignment)


@njit(parallel = False)
def update_centroids(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts):
    n_samples, n_features = samples.shape
    n_clusters = centroids.shape[0]
    n_sample_weights = sample_weights.shape[0]
   
    for i in range(n_clusters):
        centroid_counts[i] = 0.0
        for j in range(n_features):
            centroid_sums[i,j] = 0.0
            centroids[i,j] = np.nan
    
    if n_sample_weights > 0:
        for i in range(n_samples):
            centroid_ind = sample_membership[i]
            for j in range(n_features):
                centroid_sums[centroid_ind,j] += sample_weights[i] * samples[i,j]               
            centroid_counts[centroid_ind] += sample_weights[i]
    else:
        for i in range(n_samples):
            centroid_ind = sample_membership[i]
            for j in range(n_features):
                centroid_sums[centroid_ind,j] += samples[i,j]
            centroid_counts[centroid_ind] += 1.0

    for i in range(n_clusters):
        if centroid_counts[i] > 0.0:
            for j in range(n_features):
                centroids[i,j] = centroid_sums[i,j] / centroid_counts[i]
                
                

@njit(parallel = True)
def update_centroids_parallel(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts):
    n_samples, n_features = samples.shape
    n_clusters = centroids.shape[0]
    n_sample_weights = sample_weights.shape[0]
                
    for i in range(n_clusters):
        centroid_counts[i] = 0.0
        for j in range(n_features):
            centroid_sums[i,j] = 0.0
    
    thread_ranges = sub_sections(n_samples, nb.config.NUMBA_NUM_THREADS)
    n_threads = thread_ranges.shape[0]
    thread_centroid_sums = np.zeros((n_threads,n_clusters,n_features))
    thread_centroid_counts = np.zeros((n_threads,n_clusters))
    
    if n_sample_weights > 0:
        for i in prange(n_threads):
            for j in range(thread_ranges[i,0],thread_ranges[i,1]):
                centroid_ind = sample_membership[j]
                for k in range(n_features):
                    thread_centroid_sums[i,centroid_ind,k] += sample_weights[j] * samples[j,k]
                thread_centroid_counts[i,centroid_ind] += sample_weights[j]
    else:
        for i in prange(n_threads):
            for j in range(thread_ranges[i,0],thread_ranges[i,1]):
                centroid_ind = sample_membership[j]
                for k in range(n_features):
                    thread_centroid_sums[i,centroid_ind,k] += samples[j,k]
                thread_centroid_counts[i,centroid_ind] += 1.0
        
    for i in range(n_threads):
        for j in range(n_clusters):
            centroid_counts[j] += thread_centroid_counts[i,j]
            for k in range(n_features):
                centroid_sums[j,k] += thread_centroid_sums[i,j,k]

    for i in range(n_clusters):
        if centroid_counts[i] > 0.0:
            for j in range(n_features):
                centroids[i,j] = centroid_sums[i,j] / centroid_counts[i]
        else:
            for j in range(n_features):
                centroids[i,j] = np.nan
                centroid_sums[i,j] = np.nan

                


@njit
def k_means(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, max_iters = 300, tol=0.0001, parallel = True):
    n_samples, n_features = samples.shape
    n_clusters = centroids.shape[0]
        
    check_shapes(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives)  
    
    objective = np.inf
    n_iters = 0
        
    sample_membership.fill(-1)
    sample_objectives.fill(np.nan)
    centroid_objectives.fill(np.nan)
        
    if (n_samples > 0) and (n_features > 0) and (n_clusters > 0):
                
        n_changed_membership = 1
        objective_previous = np.inf
        tolerance = np.inf
                  
        while True:
                        
            if parallel:
                objective, n_changed_membership = assignment_parallel(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_objectives)
            else:
                objective, n_changed_membership = assignment(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_objectives)
            
            tolerance = 1 - objective/objective_previous
            objective_previous = objective    
                                
            n_iters += 1
            
            if (n_iters >= max_iters) or (n_changed_membership <= 0) or (tolerance <= tol) or (objective <= 0.0):
                break
                
            if parallel:
                update_centroids_parallel(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts)
            else:
                update_centroids(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts)
            
    return objective, n_iters



@njit
def k_h_means(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, k_max_iters = 300, h_max_iters = 300, k_tol=0.0001, h_tol=0.0001):
    
    k_objective, k_iters = k_means(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, k_max_iters, k_tol, True)
    
    update_centroids_parallel(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts)
    
    h_objective, h_iters = h_means_first(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts, centroid_objectives, k_objective, h_max_iters, h_tol)
           
    return h_objective, k_iters+h_iters


      
# Local search heuristic for solving the minimum sum of squares clustering problem by iteratively
# new extra center insertion, searching and worst centroid deletion. New center insertions are made 
# by using k-means++ logic. Worst center for deletion identified by its largest objective value.
@njit(parallel = True)
def iterative_extra_center_insertion_deletion(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, local_max_iters = 300, local_tol=0.0001, max_iters = 300, tol=0.0001, max_cpu_time=10, n_candidates=3, printing=False):
    n_samples, n_features = samples.shape
    n_sample_weights, = sample_weights.shape
    n_centers = centroids.shape[0]
       
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
    
    n_iters = 0
    n_local_iters = 0
    cpu_time = 0.0
    
    if (n_samples > 0) and (n_features > 0) and (n_centers > 0):
        n_centers_ext = n_centers + 1
        
        ext_sample_membership = np.full(n_samples, -1)
        ext_sample_objectives = np.full(n_samples, np.nan)        
        ext_centroids = np.full((n_centers_ext, n_features), np.nan)
        ext_centroid_sums = np.full((n_centers_ext, n_features), np.nan)
        ext_centroid_counts = np.full(n_centers_ext, 0.0)
        ext_centroid_objectives = np.full(n_centers_ext, np.nan)
                
        best_sample_membership = np.full(n_samples, -1)
        best_sample_objectives = np.full(n_samples, np.nan)
        best_centroids = np.full((n_centers_ext, n_features), np.nan)
        best_centroid_sums = np.full((n_centers_ext, n_features), np.nan)
        best_centroid_counts = np.full(n_centers_ext, 0.0)
        best_centroid_objectives = np.full(n_centers_ext, np.nan)
        
        best_objective = objective
        best_excess_centroid_ind = -1
        for i in prange(n_samples):
            best_sample_membership[i] = sample_membership[i]
            best_sample_objectives[i] = sample_objectives[i]
        for i in range(n_centers):
            best_centroid_counts[i] = centroid_counts[i]
            best_centroid_objectives[i] = centroid_objectives[i]            
            for j in range(n_features):
                best_centroids[i,j] = centroids[i,j]
                best_centroid_sums[i,j] = centroid_sums[i,j]
                                         
        n_iters = 0
        tolerance = np.inf
        
        cumsum = np.empty(n_centers_ext)
        
        with objmode(current_time = 'float64'):
            current_time = time.perf_counter()        
        cpu_time = current_time - start_time
        
        if printing: 
            with objmode:
                print ('%-30s%-15s%-15s' % ('objective', 'n_iters', 'cpu_time'))            
            
        
        while (cpu_time < max_cpu_time) and (n_iters < max_iters) and (tolerance > tol):
            
            for i in prange(n_samples):
                ext_sample_membership[i] = best_sample_membership[i]                
            for i in range(n_centers_ext):
                for j in range(n_features):
                    ext_centroids[i,j] = best_centroids[i,j]
                                    
            degenerate_mask = ext_centroid_counts == 0.0
                
            n_degenerate = np.sum(degenerate_mask)
            
            new_center_inds = additional_centers(samples, sample_weights, centroids, n_degenerate, n_candidates, distance_measure=0)
            
            ext_centroids[degenerate_mask,:] = samples[new_center_inds,:]

            ext_objective, ext_n_iters = k_means(samples, sample_weights, ext_sample_membership, ext_sample_objectives, ext_centroids, ext_centroid_sums, ext_centroid_counts, ext_centroid_objectives, local_max_iters, local_tol, True)
            
            cum_sum(ext_centroid_objectives, cumsum)            
            excess_centroid_ind = random_choice(cumsum)                        
            for i in range(n_features):
                ext_centroids[excess_centroid_ind,i] = np.nan

            ext_objective, ext_n_iters = k_means(samples, sample_weights, ext_sample_membership, ext_sample_objectives, ext_centroids, ext_centroid_sums, ext_centroid_counts, ext_centroid_objectives, local_max_iters, local_tol, True)
                                    
            with objmode(current_time = 'float64'):
                current_time = time.perf_counter()
            cpu_time = current_time - start_time                
                
            if ext_objective < best_objective:
                
                tolerance = 1 - ext_objective/best_objective
                
                best_objective = ext_objective
                for i in prange(n_samples):
                    best_sample_membership[i] = ext_sample_membership[i]
                    best_sample_objectives[i] = ext_sample_objectives[i]
                for i in range(n_centers_ext):
                    best_centroid_counts[i] = ext_centroid_counts[i]
                    best_centroid_objectives[i] = ext_centroid_objectives[i]
                    for j in range(n_features):
                        best_centroids[i,j] = ext_centroids[i,j]
                        best_centroid_sums[i,j] = ext_centroid_sums[i,j]
                best_excess_centroid_ind = excess_centroid_ind
                
                n_local_iters = ext_n_iters
                                
                if printing:
                    with objmode:
                        print ('%-30f%-15i%-15.2f' % (best_objective, n_iters, cpu_time))
                
            n_iters += 1
            
        
        if best_excess_centroid_ind > -1:
            objective = best_objective
            for i in prange(n_samples):
                sample_objectives[i] = best_sample_objectives[i]
                if best_sample_membership[i] <= best_excess_centroid_ind:
                    sample_membership[i] = best_sample_membership[i]
                else:
                    sample_membership[i] = best_sample_membership[i]-1            
            for i in range(n_centers_ext):
                if i <= best_excess_centroid_ind:
                    ind = i
                else:
                    ind = i-1
                if i != best_excess_centroid_ind:
                    centroid_counts[ind] = best_centroid_counts[i]
                    centroid_objectives[ind] = best_centroid_objectives[i]
                    for j in range(n_features):
                        centroids[ind,j] = best_centroids[i,j]
                        centroid_sums[ind,j] = best_centroid_sums[i,j]
                        
    if printing:
        with objmode:
            print ('%-30f%-15i%-15.2f' % (best_objective, n_iters, cpu_time))
                           
    return objective, n_iters, n_local_iters
       
        
@njit
def empty_solution(sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives):
    empty_sample_membership = np.empty_like(sample_membership)
    empty_sample_objectives = np.empty_like(sample_objectives)
    empty_centroids = np.empty_like(centroids)
    empty_centroid_sums = np.empty_like(centroid_sums)
    empty_centroid_counts = np.empty_like(centroid_counts)
    empty_centroid_objectives = np.empty_like(centroid_objectives)
    return empty_sample_membership, empty_sample_objectives, empty_centroids, empty_centroid_sums, empty_centroid_counts, empty_centroid_objectives


@njit(parallel = True)
def copy_solution(sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, copy_sample_membership, copy_sample_objectives, copy_centroids, copy_centroid_sums, copy_centroid_counts, copy_centroid_objectives):    
    for i in prange(sample_membership.shape[0]):
        copy_sample_membership[i] = sample_membership[i]
        copy_sample_objectives[i] = sample_objectives[i]
    for i in range(centroids.shape[0]):
        copy_centroid_counts[i] = centroid_counts[i]
        copy_centroid_objectives[i] = centroid_objectives[i]
        for j in range(centroids.shape[1]):
            copy_centroids[i,j] = centroids[i,j]
            copy_centroid_sums[i,j] = centroid_sums[i,j]    


@njit
def shake_membership(n_reallocations, n_samples, n_clusters, sample_membership):
    for i in range(n_reallocations):
        sample_membership[np.random.randint(n_samples)] = np.random.randint(n_clusters)
            

# Попробовать сделать так чтобы принимелись не любые даже сколько-нибудь малые улучшения,
# а только значимые улучшения, т.е. выше определённого порога (для этого ввести соответствующий дополнительный параметр)
@njit(parallel = True)
def Membership_Shaking_VNS(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, k_max_iters=300, h_max_iters=300, k_tol=0.0001, h_tol=0.0001, kmax=5, max_cpu_time=10, max_iters=100, printing=False):
    n_samples, n_features = samples.shape
    n_centers = centroids.shape[0]
    cpu_time = 0.0
    n_iters = 0
    k = 1
    n_iters_k = 0
    if printing: 
        with objmode:
            print ('%-30s%-7s%-15s%-15s%-15s' % ('objective', 'k', 'n_iters', 'n_iters_k', 'cpu_time'))
    
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
        
    best_sample_membership, best_sample_objectives, best_centroids, best_centroid_sums, best_centroid_counts, best_centroid_objectives = empty_solution(sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives)
    
    best_objective = objective
    
    copy_solution(sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, best_sample_membership, best_sample_objectives, best_centroids, best_centroid_sums, best_centroid_counts, best_centroid_objectives)
            
    with objmode(current_time = 'float64'):
        current_time = time.perf_counter()
                
    cpu_time = current_time- start_time       
        
    while (cpu_time < max_cpu_time) and (n_iters < max_iters):
        
        # neighborhood solution
        for i in prange(n_samples):
            sample_membership[i] = best_sample_membership[i]
                
        shake_membership(k, n_samples, n_centers, sample_membership)
        
        objective = initialize_by_membership(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives)
        
        objective, n_local_iters = k_h_means(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, k_max_iters, h_max_iters, k_tol, h_tol)
        
        with objmode(current_time = 'float64'):
            current_time = time.perf_counter()
        cpu_time = current_time - start_time
        
        if objective < best_objective:
            best_objective = objective
            
            copy_solution(sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, best_sample_membership, best_sample_objectives, best_centroids, best_centroid_sums, best_centroid_counts, best_centroid_objectives)
                        
            if printing:
                with objmode:
                    print ('%-30f%-7i%-15i%-15i%-15.2f' % (best_objective, k, n_iters, n_iters_k, cpu_time))
            k = 0
            
        k += 1
        if k > kmax:
            k = 1
            n_iters_k += 1
        n_iters += 1
                    
    objective = best_objective
    
    copy_solution(best_sample_membership, best_sample_objectives, best_centroids, best_centroid_sums, best_centroid_counts, best_centroid_objectives, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives)
    
    if printing:
        with objmode:
            print ('%-30f%-7i%-15i%-15i%-15.2f' % (best_objective, k, n_iters, n_iters_k, cpu_time))
    
    return objective, n_iters

    
# The parameter "shaking_mode" is used to define the used logic of centroid-to-entity reallocations:
# 0 - "Lumped mode" - finding a center with the worst objective and k-1 closest to it other centers, then replace these k centers with new ones (like in J-means);
# 1 - "Scatter mode" - finding a k centers distributed in the space and not connected to each other, then replace them with new ones (like in J-means);
# 2 - finding the k centers with the worst objectives and replace each of them with a random internal entity from their corresponding clusters (like I-means in [Nilolaev and Mladenovic 2015])
# 3 - replace all nondegenerate centers with a random internal entity from their corresponding clusters
@njit
def shake_centers(k, samples, sample_weights, sample_membership, centroids, centroid_counts, centroid_objectives, n_candidates=3, shaking_mode=1, fully_random = False):
    n_samples, n_features = samples.shape
    n_sample_weights, = sample_weights.shape
    n_centers = centroids.shape[0]   
    
    degenerate_mask = np.sum(np.isnan(centroids), axis = 1) > 0
    nondegenerate_mask = ~degenerate_mask
    
    n_degenerate = np.sum(degenerate_mask)
    n_nondegenerate = n_centers-n_degenerate
        
    if (k > 0) and (n_nondegenerate > 0):
        
        if k > n_nondegenerate:
            k = n_nondegenerate
        
        nondegenerate_inds = np.arange(n_centers)[nondegenerate_mask]
        nondegenerate_objectives = centroid_objectives[nondegenerate_mask]
        sum_of_centroid_objectives = np.sum(nondegenerate_objectives)
        
        if shaking_mode == 0:
            
            if fully_random:
                target = nondegenerate_inds[np.random.randint(nondegenerate_inds.shape[0])]               
            else:
                rand_val = np.random.random_sample(1) * sum_of_centroid_objectives
                target_ind = np.full(1, -1)
                cum_search(nondegenerate_objectives, rand_val, target_ind)
                target = nondegenerate_inds[target_ind[0]]

            if target > -1:            
                if k-1 > 0:
                    centroid_weights = np.empty(0)
                    dists = distance_matrix_euclidean2_XY_weighted_cpu(centroids[np.array([target])], centroids[nondegenerate_mask], centroid_weights, centroid_weights)[0]
                    replaced_inds = np.argsort(dists)[:k]
                else:
                    replaced_inds = np.full(1, target)
                    
            centroids[nondegenerate_inds[replaced_inds],:] = np.nan
            if fully_random:
                additional_center_inds = np.random.choice(n_samples, k, replace=False)
            else:    
                additional_center_inds = additional_centers(samples, sample_weights, centroids, k, n_candidates, distance_measure=0)
            centroids[nondegenerate_inds[replaced_inds],:] = samples[additional_center_inds,:]                    
        elif shaking_mode == 1:
            if fully_random:
                replaced_inds = nondegenerate_inds[np.random.choice(nondegenerate_inds.shape[0], k, replace=False)]
            else:
                rand_vals = np.random.random_sample(k) * sum_of_centroid_objectives
                replaced_inds = np.full(k, -1)
                cum_search(nondegenerate_objectives, rand_vals, replaced_inds)
            
            centroids[nondegenerate_inds[replaced_inds],:] = np.nan
            if fully_random:
                additional_center_inds = np.random.choice(n_samples, k, replace=False)
            else:
                additional_center_inds = additional_centers(samples, sample_weights, centroids, k, n_candidates, distance_measure=0)
            centroids[nondegenerate_inds[replaced_inds],:] = samples[additional_center_inds,:]
        elif shaking_mode == 2:
            if fully_random:
                replaced_inds = nondegenerate_inds[np.random.choice(nondegenerate_inds.shape[0], k, replace=False)]
            else:            
                rand_vals = np.random.random_sample(k) * sum_of_centroid_objectives
                replaced_inds = np.full(k, -1)
                cum_search(nondegenerate_objectives, rand_vals, replaced_inds)
            
            sample_inds = np.arange(n_samples)
            target_inds = nondegenerate_inds[replaced_inds]
            for i in range(k):
                candidate_inds = sample_inds[sample_membership == target_inds[i]]
                sample_ind = candidate_inds[np.random.randint(candidate_inds.shape[0])]
                centroids[target_inds[i],:] = samples[sample_ind,:]
        elif shaking_mode == 3:
            sample_inds = np.arange(n_samples)
            for i in nondegenerate_inds:
                candidate_inds = sample_inds[sample_membership == i]
                sample_ind = candidate_inds[np.random.randint(candidate_inds.shape[0])]
                centroids[i,:] = samples[sample_ind,:]
        else:
            raise KeyError
    if n_degenerate > 0:
        additional_center_inds = additional_centers(samples, sample_weights, centroids, n_degenerate, n_candidates, distance_measure=0)
        centroids[degenerate_mask,:] = samples[additional_center_inds,:]
            
    
            
# Simple center shaking VNS
@njit(parallel = True)
def Center_Shaking_VNS(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, local_max_iters=300, local_tol=0.0001, kmax=3, max_cpu_time=10, max_iters=100, n_candidates=3, shaking_mode = 0, fully_random=False, printing=False):
    n_samples, n_features = samples.shape
    n_sample_weights, = sample_weights.shape
    n_centers = centroids.shape[0]

    cpu_time = 0.0
    n_iters = 0
    k = 1
    n_iters_k = 0
    if printing: 
        with objmode:
            print ('%-30s%-7s%-15s%-15s%-15s' % ('objective', 'k', 'n_iters', 'n_iters_k', 'cpu_time'))
        
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
                
    best_objective = objective
    best_n_local_iters = 0
    
    if (n_samples > 0) and (n_features > 0) and (n_centers > 0):
        
        # Empty Neighborhood Solution
        neighborhood_sample_membership = np.full(n_samples, -1)
        neighborhood_sample_objectives = np.full(n_samples, np.nan)
        neighborhood_centroids = np.full((n_centers, n_features), np.nan)
        neighborhood_centroid_sums = np.full((n_centers, n_features), np.nan)
        neighborhood_centroid_counts = np.full(n_centers, 0.0)
        neighborhood_centroid_objectives = np.full(n_centers, np.nan)
        
        # Empty Best Solution
        best_sample_membership = np.full(n_samples, -1)
        best_sample_objectives = np.full(n_samples, np.nan)
        best_centroids = np.full((n_centers, n_features), np.nan)
        best_centroid_sums = np.full((n_centers, n_features), np.nan)
        best_centroid_counts = np.full(n_centers, 0.0)
        best_centroid_objectives = np.full(n_centers, np.nan)

        # Best Solution is the Current One
        for i in prange(n_samples):
            best_sample_membership[i] = sample_membership[i]
            best_sample_objectives[i] = sample_objectives[i]
        for i in range(n_centers):
            best_centroid_counts[i] = centroid_counts[i]
            best_centroid_objectives[i] = centroid_objectives[i]
            for j in range(n_features):
                best_centroids[i,j] = centroids[i,j]
                best_centroid_sums[i,j] = centroid_sums[i,j]

        with objmode(current_time = 'float64'):
            current_time = time.perf_counter()
        cpu_time = current_time - start_time       
                
        k = 1
        while (cpu_time < max_cpu_time) and (n_iters < max_iters):
            
            # Neighborhood Solution
            for i in prange(n_samples):
                neighborhood_sample_membership[i] = best_sample_membership[i]
                neighborhood_sample_objectives[i] = best_sample_objectives[i]
            for i in range(n_centers):
                neighborhood_centroid_counts[i] = best_centroid_counts[i]
                neighborhood_centroid_objectives[i] = best_centroid_objectives[i]
                for j in range(n_features):
                    neighborhood_centroids[i,j] = best_centroids[i,j]
                    neighborhood_centroid_sums[i,j] = best_centroid_sums[i,j]          
            shake_centers(k, samples, sample_weights, sample_membership, neighborhood_centroids, neighborhood_centroid_counts, neighborhood_centroid_objectives, n_candidates, shaking_mode, fully_random)
            
            # Local Search Initialized by Neighborhood Solution
            neighborhood_objective, neighborhood_n_iters = k_means(samples, sample_weights, neighborhood_sample_membership, neighborhood_sample_objectives, neighborhood_centroids, neighborhood_centroid_sums, neighborhood_centroid_counts, neighborhood_centroid_objectives, local_max_iters, local_tol, True)
            
            with objmode(current_time = 'float64'):
                current_time = time.perf_counter()
            cpu_time = current_time - start_time            
            
            # Check for the Best
            if neighborhood_objective < best_objective:
                best_objective = neighborhood_objective
                best_n_local_iters = neighborhood_n_iters
                if printing:
                    with objmode:
                        print ('%-30f%-7i%-15i%-15i%-15.2f' % (best_objective, k, n_iters, n_iters_k, cpu_time))
                k = 0
                
                # Remember the Best Solution
                for i in prange(n_samples):
                    best_sample_membership[i] = neighborhood_sample_membership[i]
                    best_sample_objectives[i] = neighborhood_sample_objectives[i]
                for i in range(n_centers):
                    best_centroid_counts[i] = neighborhood_centroid_counts[i]
                    best_centroid_objectives[i] = neighborhood_centroid_objectives[i]
                    for j in range(n_features):
                        best_centroids[i,j] = neighborhood_centroids[i,j]
                        best_centroid_sums[i,j] = neighborhood_centroid_sums[i,j]
                
            k += 1
            if k > kmax: 
                k = 1
                n_iters_k += 1
            n_iters += 1            
            
        # Replace Current Solution by the Best One
        for i in prange(n_samples):
            sample_membership[i] = best_sample_membership[i]
            sample_objectives[i] = best_sample_objectives[i]
        for i in range(n_centers):
            centroid_counts[i] = best_centroid_counts[i]
            centroid_objectives[i] = best_centroid_objectives[i]
            for j in range(n_features):
                centroids[i,j] = best_centroids[i,j]
                centroid_sums[i,j] = best_centroid_sums[i,j]
                
#     if printing:
#         with objmode:
#             print ('%-30f%-7i%-15i%-15i%-15.2f' % (best_objective, k, n_iters, n_iters_k, cpu_time))
                
    return best_objective, n_iters, best_n_local_iters


# Для обработки больших данных сделать возможность обработки разряженного входного датасета??? (или кому надо тот сам на вход подаст разряженный датасет???)
# Доработать эту процедуру чтобы можно было выбрать метрику.
# Использовать k-medoids вместо k-means чтобы можно было использовать полную предрасчитанную матрицу расстояний
#
# The idea of the algorithm is inspired by:
# Likasa A., Vlassisb N., Verbeek J.J. The global k-means clustering algorithm //
# Pattern Recognition 36 (2003), pp. 451 – 461
@njit(parallel=True)
def number_of_clusters(samples, min_num=-1, max_num=-1, max_iters=300, tol=0.0001):
    n_samples = samples.shape[0]
    n_features = samples.shape[1]
    
    if min_num < 2 or min_num > n_samples:
        min_num = 2
    if max_num < 0 or max_num > n_samples:
        max_num = n_samples

    if n_samples > 0 and n_features > 0 and min_num < max_num:

        objectives = np.full(max_num, 0.0)

        used_samples = np.full(n_samples, False)
        global_centroid = np.reshape(np.sum(samples, axis=0) / n_samples, (1, samples.shape[1]))

        D = distance_matrix_euclidean2_XY_cpu(global_centroid, samples)
        medoid_ind = np.argmin(D[0])

        n_centroids = 1

        sample_weights, sample_membership, sample_objectives, centroids2, centroid_sums, centroid_counts, centroid_objectives = empty_state(n_samples, n_features, n_centroids)
        centroids2[0, :] = samples[medoid_ind, :]

        centroids = np.full((max_num, n_features), np.nan)
        centroids[0, :] = samples[medoid_ind, :]
        used_samples[medoid_ind] = True

        objectives[0], _ = assignment(samples, sample_weights, sample_membership, sample_objectives, centroids2, centroid_objectives)

        local_objectives = np.empty(n_samples)

        for i in range(1, max_num):

            local_objectives.fill(np.inf)

            for j in prange(n_samples):

                if not used_samples[j]:
                    sample_weights, sample_membership, sample_objectives, centroids2, centroid_sums, centroid_counts, centroid_objectives = empty_state(n_samples, n_features, n_centroids)

                    centroids2 = np.concatenate((centroids[:n_centroids], np.reshape(samples[j], (1, samples.shape[1]))))
                    local_objectives[j], _ = assignment(samples, sample_weights, sample_membership, sample_objectives, centroids2, centroid_objectives)

            min_ind = np.argmin(local_objectives)
            used_samples[min_ind] = True
            centroids[n_centroids, :] = samples[min_ind, :]
            objectives[n_centroids] = local_objectives[min_ind]
            n_centroids += 1

        cluster_nums = np.arange(min_num, max_num)
        drop_rates = np.empty(cluster_nums.shape[0])

        for i in range(min_num - 1, max_num - 1):

            p1 = objectives[i - 1]
            p2 = objectives[i]
            p3 = objectives[i + 1]

            d1 = abs(p1 - p2)
            d2 = abs(p2 - p3)
            
            #d1 = p1 - p2
            #d2 = p2 - p3

            if d2 != 0.0:
                drop_rates[i - min_num + 1] = d1 / d2
            else:
                drop_rates[i - min_num + 1] = 0.0

        n_clusters = cluster_nums[np.argmax(drop_rates)]
    else:
        n_clusters = -1
        cluster_nums = np.full(0, -1)
        drop_rates = np.empty(0)
        objectives = np.empty(0)      
                        

    return n_clusters, cluster_nums, drop_rates, objectives


@njit
def method_sequencing(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, method_sequence, time_sequence, max_iters_sequence, kmax_sequence, local_max_iters=300, local_tol=0.00001, n_candidates=3, shaking_mode = 0, printing=False):
    assert method_sequence.ndim == 1 and time_sequence.ndim == 1 and kmax_sequence.ndim == 1
    sequence_size = method_sequence.shape[0]
    assert time_sequence.shape[0] == sequence_size and max_iters_sequence.shape[0] == sequence_size and kmax_sequence.shape[0] == sequence_size
    methods = {0,1,2,3,4,5,6}    
    for i in range(sequence_size):
        method = method_sequence[i]
        assert method in methods
        if method == 1:
            if printing: print('H-means (first improvement strategy):')
            objective, n_iters = h_means_first(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, local_max_iters, local_tol)
            if printing: 
                print(objective)
                print()
        elif method == 2:
            if printing: print('H-means (best-improvement strategy):')
            objective, n_iters = h_means_best(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, local_max_iters, local_tol)
            if printing:
                print(objective)
                print()
        elif method == 3:
            if printing: print('K-H-means:')
            objective, n_iters = k_h_means(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, local_max_iters, local_max_iters, local_tol, local_tol)
            if printing:
                print(objective)
                print()
        elif method == 4:
            if printing: print('Center Shaking VNS:')
            objective, n_iters, n_local_iters = Center_Shaking_VNS(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, local_max_iters, local_tol, kmax_sequence[i], time_sequence[i], max_iters_sequence[i], n_candidates, shaking_mode, False, printing)
            if printing: print()
        elif method == 5:
            if printing: print('Membership Shaking VNS:')
            objective, n_iters = Membership_Shaking_VNS(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, local_max_iters, local_max_iters, local_tol, local_tol, kmax_sequence[i], time_sequence[i], max_iters_sequence[i], printing)
            if printing: print()
        elif method == 6:
            if printing: print('Extra Center Insertion/Deletion:')
            objective, n_iters, n_local_iters = iterative_extra_center_insertion_deletion(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, local_max_iters, local_tol, max_iters_sequence[i], local_tol, time_sequence[i], n_candidates)
            if printing: print()
        else:
            if printing: print('K-means:')
            objective, n_iters = k_means(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, local_max_iters, local_tol, True)
            if printing: 
                print(objective)
                print()
            
    return objective
            


# Parallel multi-portion Minimum Sum-of-Squares Clustering (MSSC)
@njit(parallel = True)
def multi_portion_mssc(samples, sample_weights, centers, method_sequence, time_sequence, max_iters_sequence, kmax_sequence, n_clusters = 3, portion_size = -1, n_portions = 3, init_method = 1, local_max_iters=300, local_tol=0.0001, n_candidates=3):
    
    n_samples, n_features = samples.shape
    n_centers, n_center_features = centers.shape
    n_sample_weights, = sample_weights.shape
    init_methods = {0,1,2}

    assert ((n_samples == n_sample_weights) or (n_sample_weights == 0))
    assert ((init_method != 2) or (n_features == n_center_features))
    assert (((init_method != 2) and (n_centers == 0)) or (n_centers == n_clusters))
    assert (portion_size == -1) or (portion_size <= n_samples)
    assert init_method in init_methods
            
    collected_centroids = np.full((0, 0, 0), np.nan)
    collected_centroid_counts = np.full((0, 0), 0.0)
    collected_centroid_objectives = np.full((0, 0), np.nan)
    collected_objectives = np.full(0, np.nan)
        
    if (n_samples > 0) and (n_features > 0) and (n_clusters > 0) and (n_portions > 0) and ((portion_size > 0) or (portion_size == -1)) and (portion_size < n_samples):
                     
        collected_centroids = np.full((n_portions, n_clusters, n_features), np.nan)
        collected_centroid_counts = np.full((n_portions, n_clusters), 0.0)
        collected_centroid_objectives = np.full((n_portions, n_clusters), np.nan)
        collected_objectives = np.full(n_portions, np.nan)
                        
        if portion_size == -1:
            p_samples = samples
            p_n_samples = n_samples
            p_sample_weights = sample_weights
        
        for i in prange(n_portions):
            if portion_size > 0:
                p_inds = np.random.choice(n_samples, portion_size, replace = False)
                p_samples = samples[p_inds]
                p_n_samples = portion_size
                if n_sample_weights > 0:
                    p_sample_weights = sample_weights[p_inds]
                else:
                    p_sample_weights = np.full(0, 0.0)

            if init_method == 1:
                collected_centroids[i] = p_samples[k_means_pp(p_samples, p_sample_weights, n_clusters, n_candidates, distance_measure=0)]
            elif init_method == 2:
                collected_centroids[i] = np.copy(centers)
            else:
                collected_centroids[i] = np.random.rand(n_clusters, n_features)
                
            p_sample_membership = np.empty_like(p_inds)
            p_sample_objectives = np.empty(p_n_samples)
            p_centroid_sums = np.empty((n_clusters, n_features))           
            
            collected_objectives[i] = method_sequencing(p_samples, p_sample_weights, p_sample_membership, p_sample_objectives, collected_centroids[i], p_centroid_sums, collected_centroid_counts[i], collected_centroid_objectives[i], np.inf, method_sequence, time_sequence, max_iters_sequence, kmax_sequence, local_max_iters, local_tol, n_candidates, False)
            
    return collected_centroids, collected_centroid_counts, collected_centroid_objectives, collected_objectives


@njit(parallel = True)
def decomposition_aggregation_mssc(samples, sample_weights, method_sequence, time_sequence, max_iters_sequence, kmax_sequence, n_clusters = 3, portion_size = -1, n_portions = 3, init_method = 1, local_max_iters=300, local_tol=0.0001, n_candidates=3, aggregation_method = 0, basis_n_init = 3):
    n_samples, n_features = samples.shape
    n_sample_weights, = sample_weights.shape
    
    final_objective = np.inf
    final_n_iters = 0
    final_sample_membership = np.full(n_samples, -1)
    final_sample_objectives = np.full(n_samples, np.nan)
    final_centroids = np.full((n_clusters,n_features), np.nan)
    final_centroid_sums = np.full((n_clusters,n_features), np.nan)
    final_centroid_counts = np.full(n_clusters, 0.0)
    final_centroid_objectives = np.full(n_clusters, np.nan)
    
    if (n_samples > 0) and (n_features > 0) and (n_portions > 0) and (portion_size > 0) and (portion_size <= n_samples) and (basis_n_init > 0):
    
        centers = np.empty((0, n_features))

        centroids, centroid_counts, centroid_objectives, objectives = multi_portion_mssc(samples, sample_weights, centers, method_sequence, time_sequence, max_iters_sequence, kmax_sequence, n_clusters, portion_size, n_portions, init_method, local_max_iters, local_tol, n_candidates)

        full_objectives = np.empty_like(objectives)
        sample_membership = np.full(0, 0)
        sample_objectives = np.full(0, 0.0)
        centroid_objectives = np.empty((n_portions, n_clusters))

        for i in prange(n_portions):
            full_objectives[i], n_changed_membership = assignment(samples, sample_weights, sample_membership, sample_objectives, centroids[i], centroid_objectives[i])
            
        if aggregation_method == 0:
            min_ind = np.argmin(full_objectives)
            final_centroids[:,:] = centroids[min_ind,:,:]
        else:

            n_basis_samples = np.sum(centroid_counts > 0.0)

            basis_samples = np.empty((n_basis_samples, n_features), dtype = samples.dtype)
            basis_weights = np.empty(n_basis_samples)

            ind = 0
            for i in range(n_portions):
                for j in range(n_clusters):
                    if centroid_counts[i,j] > 0.0:
                        basis_samples[ind] = centroids[i,j]
                        #basis_weights[ind] = centroid_objectives[i,j]*full_objectives[i]
                        #basis_weights[ind] = centroid_objectives[i,j]
                        #basis_weights[ind] = centroid_objectives[i,j]/centroid_counts[i,j] #!!!!!!!!!
                        basis_weights[ind] = full_objectives[i]
                        #basis_weights[ind] = (centroid_objectives[i,j]*full_objectives[i])/centroid_counts[i,j]
                        ind += 1
            normalization1D(basis_weights, True)

            for i in range(n_basis_samples):
                basis_weights[i] = np.exp(1-basis_weights[i])

            basis_objectives = np.empty(basis_n_init)
            basis_centroids = np.full((basis_n_init, n_clusters, n_features), np.nan)                   

            for i in prange(basis_n_init):

                basis_sample_membership = np.full(n_basis_samples, -1)
                basis_sample_objectives = np.full(n_basis_samples, np.nan)
                basis_centroid_sums = np.full((n_clusters,n_features), np.nan)
                basis_centroid_counts = np.full(n_clusters, 0.0)
                basis_centroid_objectives = np.full(n_clusters, np.nan)

                basis_centroids[i,:,:] = basis_samples[k_means_pp(basis_samples, basis_weights, n_clusters, n_candidates, distance_measure=0)][:,:]
        
                basis_objectives[i] = method_sequencing(basis_samples, basis_weights, basis_sample_membership, basis_sample_objectives, basis_centroids[i], basis_centroid_sums, basis_centroid_counts, basis_centroid_objectives, np.inf, method_sequence, time_sequence, max_iters_sequence, kmax_sequence, local_max_iters, local_tol, n_candidates, False)

            min_ind = np.argmin(basis_objectives)
            final_centroids[:,:] = basis_centroids[min_ind,:,:]                           

        final_objective = method_sequencing(samples, sample_weights, final_sample_membership, final_sample_objectives, final_centroids, final_centroid_sums, final_centroid_counts, final_centroid_objectives, np.inf, method_sequence, time_sequence, max_iters_sequence, kmax_sequence, local_max_iters, local_tol, n_candidates, False)
        
    return final_objective, final_sample_membership, final_sample_objectives, final_centroids, final_centroid_sums, final_centroid_counts, final_centroid_objectives




                     
