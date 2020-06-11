# Flaky Clustering Library v0.06
# Big MSSC (Minimum Sum-Of-Squares Clustering)

# Nenad Mladenovic, Rustam Mussabayev, Alexander Krassovitskiy
# rmusab@gmail.com

# v0.06 - 05/06/2020 - New functionality: method sequencing;
# v0.05 - 04/06/2020 - New functionality:  Simple center shaking VNS, Membership shaking VNS, Iterative extra center insertion/deletion, procedure for choosing the new n additional centers for existing ones using the k-means++ logic;
# v0.04 - 17/03/2020 - Different initialization modes are added to "Decomposition/aggregation k-means";
# v0.03 - 13/03/2020 - k-means++ is implemented;
# v0.02 - 10/03/2020 - Decomposition/aggregation k-means is implemented;
# v0.01 - 27/02/2020 - Initial release. Multiprocessing k-means is implemented.

import math
import time
import pickle
import numpy as np
import numba as nb
from numba import njit, objmode
from numba import prange
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

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
def generate_dataset1(n_features = 2, n_samples = 1000, n_clusters = 5, cluster_std = 0.1):
    true_centers = np.random.rand(n_clusters, n_features)   
    X, labels = make_blobs(n_samples=n_samples, centers=true_centers, cluster_std=cluster_std)
    N = np.concatenate((true_centers,X))
    N = normalization(N)
    true_centers = N[:n_clusters]
    X = N[n_clusters:]
    #X = X.astype(np.float32)
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



@njit(parallel=True)
def squared_distance_matrix_XY(X, Y, weightsX, weightsY,  out):
    n_rowsX = X.shape[0]
    n_colsX = X.shape[1]
    n_rowsY = Y.shape[0]
    n_weightsX = weightsX.shape[0]
    n_weightsY = weightsY.shape[0]
    if (n_weightsX > 0) and (n_weightsY > 0):
        for i in prange(n_rowsX):
            for j in range(n_rowsY):            
                out[i,j] = 0.0
                for k in range(n_colsX):
                    out[i,j] += (X[i,k]-Y[j,k])**2
                out[i,j] *= weightsX[i]*weightsY[j]
    else:
        for i in prange(n_rowsX):
            for j in range(n_rowsY):            
                out[i,j] = 0.0
                for k in range(n_colsX):
                    out[i,j] += (X[i,k]-Y[j,k])**2

                    
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
def k_means_pp_naive(samples, sample_weights, n_clusters):
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
                min_dist2 = np.inf
                for j in range(n_centroids):
                    dist2 = 0.0
                    for h in range(n_features):
                        dist2 += (samples[centroid_inds[j],h] - samples[i,h])**2
                    dist2 *= sample_weights[i]*sample_weights[centroid_inds[j]]
                    if dist2 < min_dist2:
                        min_dist2 = dist2
                if min_dist2 < np.inf:
                    weights[i] = min_dist2
            
            cum_sum(weights, cumsum)
            new_centroid = random_choice(cumsum)
                        
            centroid_inds[n_centroids] = new_centroid
            n_centroids += 1    
    return centroid_inds



@njit(parallel=True)
def additional_centers_naive(samples, sample_weights, centroids, n_additional_centers=1):
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
                min_dist2 = np.inf
                
                for j in range(n_nondegenerate_clusters):
                    dist2 = 0.0
                    for h in range(n_features):
                        dist2 += (samples[i,h]-nondegenerate_centroids[j,h])**2
                    if dist2 < min_dist2:
                        min_dist2 = dist2
                        
                for j in range(n_added_centers):
                    dist2 = 0.0
                    for h in range(n_features):
                        dist2 += (samples[i,h]-samples[center_inds[j],h])**2
                    if dist2 < min_dist2:
                        min_dist2 = dist2                                               
                        
                if min_dist2 < np.inf:
                    weights[i] = min_dist2 * sample_weights[i]
            
            cum_sum(weights, cumsum)
            new_centroid = random_choice(cumsum)
            center_inds[c] = new_centroid
            
            n_added_centers += 1
            
    return center_inds
  
    
# k-means++ : algorithm for choosing the initial cluster centers (or "seeds") for the k-means clustering algorithm
# samples должны быть хорошо перемешаны ??????? !!!!!!!!!!!!!!!!!!
@njit(parallel=True)
def k_means_pp(samples, sample_weights, n_centers=3, n_candidates=3):
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
                
        dist_mat = np.empty((1,n_samples))
        indices = np.full(1, center_inds[0])
        squared_distance_matrix_XY(samples[indices], samples, sample_weights[indices], sample_weights, dist_mat)
        
        closest_dist_sq = dist_mat[0]
        
        current_pot = 0.0
        for i in prange(n_samples):
            current_pot += closest_dist_sq[i]
                                      
        candidate_ids = np.full(n_candidates, -1)
                
        distance_to_candidates = np.empty((n_candidates,n_samples))
        candidates_pot = np.empty(n_candidates)
                
        for c in range(1,n_centers):
            rand_vals = np.random.random_sample(n_candidates) * current_pot
            
            cum_search(closest_dist_sq, rand_vals, candidate_ids)
                
            squared_distance_matrix_XY(samples[candidate_ids], samples, sample_weights[candidate_ids], sample_weights, distance_to_candidates)
            
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
def additional_centers(samples, sample_weights, centroids, n_additional_centers=3, n_candidates=3):
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
            distance_to_centroids = np.empty((n_nondegenerate_clusters, n_samples))
            
            centroid_weights = np.ones(n_nondegenerate_clusters)
            
            squared_distance_matrix_XY(centroids[nondegenerate_mask], samples, centroid_weights, sample_weights, distance_to_centroids)
                    
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

            dist_mat = np.empty((1,n_samples))
            indices = np.full(1, center_inds[0])
            squared_distance_matrix_XY(samples[indices], samples, sample_weights[indices], sample_weights, dist_mat)

            closest_dist_sq = dist_mat[0]
            
            current_pot = 0.0
            for i in prange(n_samples):
                current_pot += closest_dist_sq[i]
                
            n_added_centers = 1        
        
        candidate_ids = np.full(n_candidates, -1)
                
        distance_to_candidates = np.empty((n_candidates,n_samples))
        candidates_pot = np.empty(n_candidates)
                
        for c in range(n_added_centers, n_additional_centers):
            
            rand_vals = np.random.random_sample(n_candidates) * current_pot
            
            cum_search(closest_dist_sq, rand_vals, candidate_ids)
                
            squared_distance_matrix_XY(samples[candidate_ids], samples, sample_weights[candidate_ids], sample_weights, distance_to_candidates)
            
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



@njit(parallel = True)
def assignment(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_objectives):
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


@njit(parallel = True)
def update_centroids(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts):
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
def k_means(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, max_iters = 300, tol=0.0001):
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
                        
            objective, n_changed_membership = assignment(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_objectives)
            
            tolerance = 1 - objective/objective_previous
            objective_previous = objective    
                                
            n_iters += 1
            
            if (n_iters >= max_iters) or (n_changed_membership <= 0) or (tolerance <= tol) or (objective <= 0.0):
                break
                
            update_centroids(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts)
            
    return objective, n_iters


@njit
def k_h_means(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, k_max_iters = 600, h_max_iters = 300, k_tol=0.0001, h_tol=0.00005):
    
    k_objective, k_iters = k_means(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, k_max_iters, k_tol)
    
    update_centroids(samples, sample_weights, sample_membership, centroids, centroid_sums, centroid_counts)
    
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
        
        while (cpu_time < max_cpu_time) and (n_iters < max_iters) and (tolerance > tol):
            
            for i in prange(n_samples):
                ext_sample_membership[i] = best_sample_membership[i]                
            for i in range(n_centers_ext):
                for j in range(n_features):
                    ext_centroids[i,j] = best_centroids[i,j]
                                    
            degenerate_mask = ext_centroid_counts == 0.0
                
            n_degenerate = np.sum(degenerate_mask)
            
            new_center_inds = additional_centers(samples, sample_weights, centroids, n_degenerate, n_candidates)
            
            ext_centroids[degenerate_mask,:] = samples[new_center_inds,:]

            ext_objective, ext_n_iters = k_means(samples, sample_weights, ext_sample_membership, ext_sample_objectives, ext_centroids, ext_centroid_sums, ext_centroid_counts, ext_centroid_objectives, local_max_iters, local_tol)
            
            cum_sum(ext_centroid_objectives, cumsum)            
            excess_centroid_ind = random_choice(cumsum)                        
            for i in range(n_features):
                ext_centroids[excess_centroid_ind,i] = np.nan

            ext_objective, ext_n_iters = k_means(samples, sample_weights, ext_sample_membership, ext_sample_objectives, ext_centroids, ext_centroid_sums, ext_centroid_counts, ext_centroid_objectives, local_max_iters, local_tol)
                                    
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
                                
                if printing: print(best_objective)
                
            n_iters += 1
            
            with objmode(current_time = 'float64'):
                current_time = time.perf_counter()
            cpu_time = current_time - start_time
        
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
                        
    if printing: print(objective)
                           
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
            

@njit(parallel = True)
def Membership_Shaking_VNS(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, k_max_iters=300, h_max_iters=300, k_tol=0.0001, h_tol=0.0001, kmax=5, max_cpu_time=10, max_iters=100, printing=False):
    n_samples, n_features = samples.shape
    n_centers = centroids.shape[0]
    
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
        
    best_sample_membership, best_sample_objectives, best_centroids, best_centroid_sums, best_centroid_counts, best_centroid_objectives = empty_solution(sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives)
    
    best_objective = objective
    
    copy_solution(sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, best_sample_membership, best_sample_objectives, best_centroids, best_centroid_sums, best_centroid_counts, best_centroid_objectives)
            
    with objmode(current_time = 'float64'):
        current_time = time.perf_counter()
                
    cpu_time = current_time- start_time 
    n_iters = 0
    k = 1
    while (cpu_time < max_cpu_time) and (n_iters < max_iters):
        
        # neighborhood solution
        for i in prange(n_samples):
            sample_membership[i] = best_sample_membership[i]
                
        shake_membership(k, n_samples, n_centers, sample_membership)
        
        objective = initialize_by_membership(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives)
        
        objective, n_local_iters = k_h_means(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, k_max_iters, h_max_iters, k_tol, h_tol)
        
        if objective < best_objective:
            best_objective = objective
            
            copy_solution(sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, best_sample_membership, best_sample_objectives, best_centroids, best_centroid_sums, best_centroid_counts, best_centroid_objectives)
                        
            if printing: print(best_objective, k)
            k = 0
            
        k += 1
        if k > kmax: k = 1
        n_iters += 1
        
        with objmode(current_time = 'float64'):
            current_time = time.perf_counter()
        cpu_time = current_time - start_time
            
    objective = best_objective
    
    copy_solution(best_sample_membership, best_sample_objectives, best_centroids, best_centroid_sums, best_centroid_counts, best_centroid_objectives, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives)
    
    if printing: print(objective)
        
    return objective, n_iters
    

@njit
def shake_centers(n_reallocations, samples, sample_weights, centroids, centroid_counts, centroid_objectives, n_candidates=3):
    n_samples, n_features = samples.shape
    n_sample_weights, = sample_weights.shape
    n_centers = centroids.shape[0]
        
    if n_reallocations > 0:
                    
        sum_of_centroid_objectives = 0.0
        for i in range(n_centers):
            if not np.isnan(centroid_objectives[i]):
                sum_of_centroid_objectives += centroid_objectives[i]
        rand_vals = np.random.random_sample(n_reallocations) * sum_of_centroid_objectives
        eliminated_inds = np.full(n_reallocations, -1)
        cum_search(centroid_objectives, rand_vals, eliminated_inds)

        n_added_centers = 0
        for i in range(n_centers):
            is_eliminated = False
            for j in range(n_reallocations):
                if eliminated_inds[j] == i:
                    is_eliminated = True
                    break            
            if (not is_eliminated) and (centroid_counts[i] > 0.0):
                centroids[n_added_centers,:] = centroids[i,:]
                n_added_centers += 1
                
        centroids[n_added_centers:,:] = np.nan
        
        n_additional_centers = n_centers - n_added_centers
        
        additional_center_inds = additional_centers(samples, sample_weights, centroids, n_additional_centers, n_candidates)
            
        for i in range(n_additional_centers):
            centroids[n_added_centers,:] = samples[additional_center_inds[i],:]
            n_added_centers += 1

            
# Simple center shaking VNS
@njit(parallel = True)
def Center_Shaking_VNS(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, local_max_iters=300, local_tol=0.0001, kmax=5, max_cpu_time=10, max_iters=100, n_candidates=3, printing=False):
    n_samples, n_features = samples.shape
    n_sample_weights, = sample_weights.shape
    n_centers = centroids.shape[0]       
    
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
                
    best_objective = objective
    n_iters = 0
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
            shake_centers(k, samples, sample_weights, neighborhood_centroids, neighborhood_centroid_counts, neighborhood_centroid_objectives, n_candidates)
            
            # Local Search Initialized by Neighborhood Solution
            neighborhood_objective, neighborhood_n_iters = k_means(samples, sample_weights, neighborhood_sample_membership, neighborhood_sample_objectives, neighborhood_centroids, neighborhood_centroid_sums, neighborhood_centroid_counts, neighborhood_centroid_objectives, local_max_iters, local_tol)
            
            # Check for the Best
            if neighborhood_objective < best_objective:
                best_objective = neighborhood_objective
                best_n_local_iters = neighborhood_n_iters
                if printing: print(best_objective, k)
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
            if k > kmax: k = 1        
            n_iters += 1
            
            with objmode(current_time = 'float64'):
                current_time = time.perf_counter()
            cpu_time = current_time - start_time
            
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
                
    if printing: print(best_objective)
                
    return best_objective, n_iters, best_n_local_iters


@njit
def method_sequencing(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, method_sequence, time_sequence, max_iters_sequence, kmax_sequence, local_max_iters=300, local_tol=0.00001, n_candidates=3, printing=False):
    assert method_sequence.ndim == 1 and time_sequence.ndim == 1 and kmax_sequence.ndim == 1
    sequence_size = method_sequence.shape[0]
    assert time_sequence.shape[0] == sequence_size and kmax_sequence.shape[0] == sequence_size
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
            objective, n_iters, n_local_iters = Center_Shaking_VNS(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, objective, local_max_iters, local_tol, kmax_sequence[i], time_sequence[i], max_iters_sequence[i], n_candidates, printing)
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
            objective, n_iters = k_means(samples, sample_weights, sample_membership, sample_objectives, centroids, centroid_sums, centroid_counts, centroid_objectives, local_max_iters, local_tol)
            if printing: 
                print(objective)
                print()
            
    return objective
            


# Parallel multi-portion Minimum Sum of Squares Clustering (MSSC)
@njit(parallel = True)
def multi_portion_mssc(samples, sample_weights, centers, n_clusters = 3, portion_size = -1, n_portions = 3, init_method = 1, search_method=0, kmax=5, local_max_iters=300, local_tol=0.0001, max_cpu_time=10, max_iters=100, n_candidates=3):
    
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
                collected_centroids[i] = samples[k_means_pp(p_samples, p_sample_weights, n_clusters, n_candidates)]
            elif init_method == 2:
                collected_centroids[i] = np.copy(centers)
            else:
                collected_centroids[i] = np.random.rand(n_clusters, n_features)
                
            p_sample_membership = np.empty_like(p_inds)
            p_sample_objectives = np.empty(p_n_samples)
            p_centroid_sums = np.empty((n_clusters, n_features))
            
            if search_method == 1:
                collected_objectives[i], p_n_iters = k_means(p_samples, p_sample_weights, p_sample_membership, p_sample_objectives, collected_centroids[i], p_centroid_sums, collected_centroid_counts[i], collected_centroid_objectives[i], local_max_iters, local_tol)
                
                collected_objectives[i], p_n_iters, p_n_local_iters = Center_Shaking_VNS(p_samples, p_sample_weights, p_sample_membership, p_sample_objectives, collected_centroids[i], p_centroid_sums, collected_centroid_counts[i], collected_centroid_objectives[i], collected_objectives[i], local_max_iters, local_tol, kmax, max_cpu_time, max_iters, n_candidates)
            else:
                collected_objectives[i], p_n_iters = k_means(p_samples, p_sample_weights, p_sample_membership, p_sample_objectives, collected_centroids[i], p_centroid_sums, collected_centroid_counts[i], collected_centroid_objectives[i], local_max_iters, local_tol)
            
    return collected_centroids, collected_centroid_counts, collected_centroid_objectives, collected_objectives



@njit(parallel = True)
def decomposition_aggregation_mssc(samples, sample_weights, n_clusters = 3, portion_size = -1, n_portions = 3, init_method = 1, search_method=0, kmax=5, local_max_iters=300, local_tol=0.0001, max_cpu_time=10, max_iters=100, n_candidates=3, aggregation_method = 0, basis_n_init = 3):
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

        centroids, centroid_counts, centroid_objectives, objectives = multi_portion_mssc(samples, sample_weights, centers, n_clusters, portion_size, n_portions, init_method, search_method, kmax, local_max_iters, local_tol, max_cpu_time, max_iters, n_candidates)

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
                        basis_weights[ind] = centroid_objectives[i,j]/centroid_counts[i,j] #!!!!!!!!!
                        #basis_weights[ind] = full_objectives[i]
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

                basis_centroids[i,:,:] = basis_samples[k_means_pp(basis_samples, basis_weights, n_clusters, n_candidates)][:,:]

                basis_objectives[i], basis_n_iters = k_means(basis_samples, basis_weights, basis_sample_membership, basis_sample_objectives, basis_centroids[i], basis_centroid_sums, basis_centroid_counts, basis_centroid_objectives, local_max_iters, local_tol)

            min_ind = np.argmin(basis_objectives)
            final_centroids[:,:] = basis_centroids[min_ind,:,:]
                           
        final_objective, final_n_iters = k_means(samples, sample_weights, final_sample_membership, final_sample_objectives, final_centroids, final_centroid_sums, final_centroid_counts, final_centroid_objectives, local_max_iters, local_tol)
        
    return final_objective, final_n_iters, final_sample_membership, final_sample_objectives, final_centroids, final_centroid_sums, final_centroid_counts, final_centroid_objectives



# Dataset Chunking VNS
@njit(parallel = True)
def Chunk_Shaking_VNS(samples, sample_weights, n_clusters = 3, kmax=5, chunk_size = -1, n_chunks = 3, init_method = 1,  local_max_iters=300, local_tol=0.0001, max_cpu_time=10, max_iters=100, n_candidates=3):
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
    
    if (n_samples > 0) and (n_features > 0) and (n_chunks > 0) and (chunk_size > 0) and (chunk_size <= n_samples):
        
        if chunk_size == -1:
            chunk_size = n_samples
       
        entities = np.empty((n_chunks, chunk_size))
        entity_weights = np.empty((n_chunks, chunk_size))
        centroids = np.full((n_chunks, n_clusters, n_features), np.nan)
        centroid_counts = np.full((n_chunks, n_clusters), 0.0)
        centroid_objectives = np.full((n_chunks, n_clusters), np.nan)
        objectives = np.full(n_chunks, np.nan)
        
        for i in prange(n_chunks):
            if chunk_size > 0:
                chunk_inds = np.random.choice(n_samples, chunk_size, replace = False)
                entities[i,:] = samples[chunk_inds][:]
                if n_sample_weights > 0:
                    entity_weights = sample_weights[chunk_inds]
                else:
                    entity_weights = np.full(0, 0.0)

            if init_method == 1:
                collected_centroids[i] = samples[k_means_pp(p_samples, p_sample_weights, n_clusters, n_candidates)]
            elif init_method == 2:
                collected_centroids[i] = np.copy(centers)
            else:
                collected_centroids[i] = np.random.rand(n_clusters, n_features)
                
            p_sample_membership = np.empty_like(p_inds)
            p_sample_objectives = np.empty(p_n_samples)
            p_centroid_sums = np.empty((n_clusters, n_features))
            
            collected_objectives[i], p_n_iters = k_means(p_samples, p_sample_weights, p_sample_membership, p_sample_objectives, collected_centroids[i], p_centroid_sums, collected_centroid_counts[i], collected_centroid_objectives[i], local_max_iters, local_tol)

                
        final_objective, final_n_iters = k_means(samples, sample_weights, final_sample_membership, final_sample_objectives, final_centroids, final_centroid_sums, final_centroid_counts, final_centroid_objectives, local_max_iters, local_tol)
        
    return final_objective, final_n_iters, final_sample_membership, final_sample_objectives, final_centroids, final_centroid_sums, final_centroid_counts, final_centroid_objectives
                    
        
                    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



            
