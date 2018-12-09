"""
Based on the journal article by Yuchen Guo, Guiguang Ding, Jungong Han, and Yue Gao, 
"Zero-shot learning with transferred samples." 
Published in IEEE Transactions on Image Processing, 26(7):3277–3290, 2017.

Developed for work in CWRU's EECS 440 course taught by Dr. Soumya Ray, Fall 2018

Written by Noah Crowley
"""

import math
import random

import numpy as np
from scipy.stats import norm as gaussian
from sklearn.svm import SVC as SVM

"""
This is for brevity.
Source and target samples are expected to be from the same vector space.
The source and target labels should be non-overlapping.
At least one of the two auxiliary information fields, either embedding_vectors or class_similarities, must be provided
Beta, delta, tau, max_ranking_iters, max_outer_iters, and num_source_to_transfer are all properties that are set to default values but can be overriden
"""
class StzslArgs:
    def __init__(self, source_samples, source_labels, source_class_list, target_samples, target_class_list, embedding_vectors = None, class_similarities = None):
        if embedding_vectors is None and class_similarities is None:
            raise ValueError("Cannot have both embedding_vectors and class_similarities be None")

        self.source_samples = source_samples
        self.source_labels = source_labels
        self.source_class_list = source_class_list
        self.target_samples = target_samples or []
        self.target_class_list = target_class_list
        self.embedding_vectors = embedding_vectors
        self.class_similarities = class_similarities
        
        self.beta = 10
        self.delta = 5
        self.tau = 1.001
        self.max_ranking_iters = 5
        self.max_outer_iters = 5 if len(self.target_samples) > 0 else 1 # Only do this iteratively for the transductive setting, i.e. we have some unlabeled target samples

        self.num_source_to_transfer = math.ceil(len(self.source_samples) / 5)

"""
Stands for "Sample Transfer Zero-Shot Learning"
Takes in an StzslArgs instance.
Returns N^t one-vs-all sklearn.svm.SVC classifiers, one for each target label
"""
def stzsl(args):
    all_samples = args.source_samples + args.target_samples

    source_class_classifiers = _get_source_class_classifiers(args, all_samples)

    sigma = _compute_mean_distance_of_vectors(all_samples)

    source_samples_similarity_matrix = _compute_samples_similarity_matrix(sigma, args.source_samples)
    target_samples_similarity_matrix = _compute_samples_similarity_matrix(sigma, args.target_samples)

    source_sample_transferability_values_by_target_class = _get_transferability_values_by_target_class(args, sigma, args.source_samples, args.source_labels, source_class_classifiers)
    transferred_source_sample_one_vs_all_labellings = _get_transferred_one_vs_all_labellings(args, sigma, args.source_samples, args.source_labels, source_sample_transferability_values_by_target_class, source_samples_similarity_matrix, args.num_source_to_transfer)

    target_class_classifiers = _build_target_classifiers(args, all_samples, transferred_source_sample_one_vs_all_labellings, [[0] * len(args.target_samples) for i in range(len(args.target_class_list))], source_sample_transferability_values_by_target_class, [[0] * len(args.target_samples) for i in range(len(args.target_class_list))])
    target_sample_transferability_values_by_target_class = None # Will be used to serve as weights, start as None

    num_iters = 1 # Already did the base iteration
    while num_iters < args.max_outer_iters:
        target_sample_pseudo_labels = _get_target_samples_pseudo_labels(args, all_samples, source_class_classifiers, target_class_classifiers)
        target_sample_transferability_values_by_target_class = _get_transferability_values_by_target_class(args, sigma, args.target_samples, target_sample_pseudo_labels, source_class_classifiers, target_class_classifiers, sample_weights_by_target_class=target_sample_transferability_values_by_target_class)
        num_to_transfer = math.ceil(len(args.target_samples) / len(args.target_class_list))
        transferred_target_sample_one_vs_all_labellings = _get_transferred_one_vs_all_labellings(args, sigma, args.target_samples, target_sample_pseudo_labels, target_sample_transferability_values_by_target_class, target_samples_similarity_matrix, num_to_transfer)
        target_class_classifiers = _build_target_classifiers(args, all_samples, transferred_source_sample_one_vs_all_labellings, transferred_target_sample_one_vs_all_labellings, source_sample_transferability_values_by_target_class, target_sample_transferability_values_by_target_class)
        num_iters += 1

    return target_class_classifiers

def _get_source_class_classifiers(args, all_samples):
    source_class_classifiers = [] # To find a mapping between classifier and label, just use source_class_classifiers[i] and source_class_list[i]
    for source_class in args.source_class_list:
        one_vs_all_labels = [0] * len(all_samples)
        for i in range(len(args.source_samples)):
            one_vs_all_labels[i] = 1 if args.source_labels[i] == source_class else 0
        classifier = SVM(kernel="linear")
        classifier.fit(all_samples, one_vs_all_labels)
        source_class_classifiers.append(classifier)
    return source_class_classifiers

def _compute_mean_distance_of_vectors(samples):
    total_distance = 0
    for i in range(len(samples)):
        for j in range(len(samples)):
            total_distance += np.linalg.norm(samples[i] - samples[j])
    return total_distance / len(samples)**2

def _compute_samples_similarity_matrix(sigma, samples):
    samples_similarity_matrix = [[0] * len(samples) for i in range(len(samples))]
    for i in range(len(samples)):
        for j in range(len(samples)):
            samples_similarity_matrix[i][j] = np.exp(-(np.linalg.norm(samples[i] - samples[j])**2) / (sigma**2))
    return np.asarray(samples_similarity_matrix)

def _get_target_samples_pseudo_labels(args, all_samples, source_class_classifiers, target_class_classifiers):
    target_sample_pseudo_labels = [-1] * len(args.target_samples) # Use -1 as no class is expected to ever be -1

    for i in range(len(target_class_classifiers)): # Preference for target classes
        predictions = target_class_classifiers[i].predict(all_samples)
        for j in range(len(args.source_samples) + 1, len(predictions)):
            jt = j - len(args.source_samples)
            if predictions[j] > 0 and target_sample_pseudo_labels[jt] == -1:
                target_sample_pseudo_labels[jt] = args.target_class_list[i]

    for i in range(len(source_class_classifiers)):
        predictions = source_class_classifiers[i].predict(args.target_samples)
        for j in range(len(predictions)):
            if predictions[j] > 0 and target_sample_pseudo_labels[j] == -1:
                target_sample_pseudo_labels[j] = args.target_class_list[i]

    return target_sample_pseudo_labels
    
def _get_sample_classification_embedding_vectors(args, samples, labels, sample_weights):
    sample_classification_embedding_vectors = []

    for i in range(len(samples)):
        sample_classification_embedding_vectors.append(args.embedding_vectors[labels[i]] * sample_weights[i])

    return sample_classification_embedding_vectors
    
def _get_transferability_from_embedding_vectors(args, sigma, samples, labels, sample_weights, target_class):
    transferability_values = [0] * len(samples)

    samples_matrix = np.asarray(samples) * np.asarray(sample_weights)[:, np.newaxis]
    sample_classification_embedding_vectors = np.asarray(_get_sample_classification_embedding_vectors(args, samples, labels, sample_weights))

    d = len(samples[0])
    x_squared = np.matmul(samples_matrix.T, samples_matrix)
    identity_d = np.identity(d)
    eps = 0.0001
    xt_a = np.matmul(samples_matrix.T, sample_classification_embedding_vectors)
    projection_matrix = np.matmul(np.linalg.inv(x_squared + identity_d*eps), xt_a) # Eq. 4

    variance = sigma**2
    for i in range(len(samples)):
        projected_sample = np.matmul(samples[i], projection_matrix)
        distance_from_embedding_vector = np.linalg.norm(args.embedding_vectors[target_class] - projected_sample)
        transferability_values[i] = gaussian.pdf(distance_from_embedding_vector, loc=0, scale=variance)

    return transferability_values

def _get_classifier_results(args, source_class_classifiers, target_class_classifiers=[]):
    total_classifier_results = [None] * (len(args.source_class_list) + len(args.target_class_list))

    all_samples = args.source_samples + args.target_samples

    for source_class_index in range(len(source_class_classifiers)):
        classifier_results = (source_class_classifiers[source_class_index].predict(all_samples) + 1) / 2
        total_classifier_results[args.source_class_list[source_class_index]] = classifier_results

    for target_class_index in range(len(target_class_classifiers)):
        classifier_results = (target_class_classifiers[target_class_index].predict(all_samples) + 1) / 2
        total_classifier_results[args.target_class_list[target_class_index]] = classifier_results

    return total_classifier_results

def _get_transferability_from_class_similarity_matrix(args, sigma, samples, labels, source_class_classifiers, target_class_classifiers, target_class):
    transferability_values = [0] * len(samples)

    classifier_results = _get_classifier_results(args, source_class_classifiers, target_class_classifiers)

    # Only use target class if in the transductive setting (i.e. there are some unlabeled target samples)
    class_list = args.source_class_list + args.target_class_list if len(target_class_classifiers) > 0 else args.source_class_list

    for i in range(len(samples)):
        # Calculate p_tilde according to Eq. 6
        p_tilde = [0] * len(class_list)
        p_tilde_sum = 0
        for c in range(len(class_list)):
            p_tilde[c] = np.exp(classifier_results[class_list[c]][i])
            p_tilde_sum += p_tilde[c]
        for c in range(len(class_list)):
            if p_tilde_sum == 0:
                p_tilde[c] = 0
            else:
                p_tilde[c] /= p_tilde_sum
            
        # Calculate transferability_value according to Eq. 7
        # Don't need the normalization by Z
        transferability_value = 0
        for c in range(len(class_list)):
            transferability_value += p_tilde_sum * args.class_similarities[target_class][class_list[c]]
        transferability_values[i] = transferability_value

    return transferability_values / np.sum(transferability_values)

def _get_transferability_values_by_target_class(args, sigma, samples, labels, source_class_classifiers, target_class_classifiers=[], sample_weights_by_target_class=None):
    transferability_values_by_target_class = []

    for i in range(len(args.target_class_list)):
        target_class = args.target_class_list[i]
        if args.embedding_vectors is not None:
            sample_weights = None
            if sample_weights_by_target_class is not None:
                sample_weights = sample_weights_by_target_class[i]
            else:
                sample_weights = [1] * len(samples)
            
            transferability_values_by_target_class.append(_get_transferability_from_embedding_vectors(args, sigma, samples, labels, sample_weights, target_class))
        else:
            transferability_values_by_target_class.append(_get_transferability_from_class_similarity_matrix(args, sigma, samples, labels, source_class_classifiers, target_class_classifiers, target_class))

    return transferability_values_by_target_class

# Based on Eq. 14
def _calculate_auxiliary(r, eta_1, mu):
    u = np.zeros_like(r)
    for i in range(len(u)):
        u[i] = max(np.zeros_like(r[i]), r[i] + (eta_1[i] / mu))
    return u

# Based on pseudo-code from Algorithm 1
def _compute_transferability_ranking_scores(transferability_values, samples_similarity_matrix, beta, tau, max_num_iters):
    transferability_values = np.asarray(transferability_values)

    # First define my parameter-dependent constants
    n = len(transferability_values)
    beta_k = samples_similarity_matrix * beta
    identity_nxn = np.identity(n)
    ones_row = np.ones((n))
    ones_sqr = np.ones((n, n))

    # Now initialize my variables
    mu = random.random() + 0.001 #Plus 0.001 to avoid starting at 0
    rankings = transferability_values / np.linalg.norm(transferability_values)
    auxiliary = np.copy(rankings)
    eta_1 = np.zeros_like(transferability_values)
    eta_2 = 0

    #Now update them
    num_iters = 0
    while num_iters < max_num_iters:
        rho = np.sum(rankings) # Eq. 8
        a = beta_k + (mu * (identity_nxn + ones_sqr)) # Algo 1, line 3
        b = transferability_values + (mu * rho) + (mu * auxiliary) - eta_1 - (eta_2 * ones_row) # Algo 1, line 4
        rankings = np.linalg.solve(a, b) # Algo 1, line 5
        auxiliary = _calculate_auxiliary(rankings, eta_1, mu) # Algo 1, line 6
        eta_1 += mu * (rankings - auxiliary) # Eq. 15; Algo 1, line 7
        eta_2 += mu * (ones_row * rankings - rho) # Eq. 15; Algo 1, line 7
        mu *= tau # Eq. 15; Algo 1, line 7
        num_iters += 1

    return rankings

def _get_transferred_one_vs_all_labellings(args, sigma, samples, labels, transferability_values_by_target_class, samples_similarity_matrix, num_to_transfer):
    transferred_one_vs_all_labellings = [[0] * len(samples) for i in range(len(args.target_class_list))]

    for i in range(len(args.target_class_list)):
        transferability_rankings = _compute_transferability_ranking_scores(
            transferability_values_by_target_class[i],
            samples_similarity_matrix,
            args.beta,
            args.tau,
            args.max_ranking_iters
        )

        most_transferrable_indices = np.argpartition(transferability_rankings, -1 * num_to_transfer)[-1 * num_to_transfer:]
        for sample_index in most_transferrable_indices:
            transferred_one_vs_all_labellings[i][sample_index] = 1
    
    return transferred_one_vs_all_labellings

def _compute_target_classifier_kernel_matrix_entry(i, j, args, samples, one_vs_all_labels, transferability_values):
    m_expectation_entry = 1 # if i == j
    if i != j:
        theta_i = 0
        if one_vs_all_labels[i] == 1:
            theta_i = 1 / (1 + np.exp(args.delta * transferability_values[i])) # Eq. 18

        theta_j = 0
        if one_vs_all_labels[j] == 1:
            theta_j = 1 / (1 + np.exp(args.delta * transferability_values[j])) # Eq. 18
        
        m_expectation_entry = 1 - 2 * (theta_i + theta_j) + (4 * theta_i * theta_j) # Eq. 19

    return np.dot(samples[i], samples[j]) * m_expectation_entry

def _compute_target_classifier_kernel_matrix(args, samples, one_vs_all_labels, transferability_values):
    k = np.zeros((len(samples), len(samples)))
    for i in range(len(samples)):
        for j in range(len(samples)):
            entry = _compute_target_classifier_kernel_matrix_entry(i, j, args, samples, one_vs_all_labels, transferability_values)
            k[i][j] = entry
    return k

def _build_target_classifier(args, samples, one_vs_all_labels, transferability_values):
    compute_kernel = lambda x, y: _compute_target_classifier_kernel_matrix(args, samples, one_vs_all_labels, transferability_values)
    classifier = SVM(kernel=compute_kernel)
    classifier.fit(samples, one_vs_all_labels)
    return classifier

def _build_target_classifiers(args, samples, transferred_source_sample_one_vs_all_labellings, transferred_target_sample_one_vs_all_labellings, source_sample_transferability_values_by_target_class, target_sample_transferability_values_by_target_class):
    target_classifiers = []

    for i in range(len(args.target_class_list)):
        one_vs_all_labels = transferred_source_sample_one_vs_all_labellings[i] + transferred_target_sample_one_vs_all_labellings[i]
        transferability_values = source_sample_transferability_values_by_target_class[i] + target_sample_transferability_values_by_target_class[i]
        target_classifiers.append(_build_target_classifier(args, samples, one_vs_all_labels, transferability_values))

    return target_classifiers
