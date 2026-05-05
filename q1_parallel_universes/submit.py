"""
Question 1: Parallel Universes Transformation
Implementation of Gradient Descent for finding the transformation matrix
and Cosine Similarity for finding the K-Nearest Neighbors.
"""

# Implement by: Alireza Sepehri
import numpy as np

def train_transformation(X, Y, R, train_steps=100, learning_rate=0.0003):
    """
    Find the best transformation matrix R mapping X to Y using Gradient Descent.
    
    Args:
        X (np.ndarray): The source universe matrix of shape (n, 300).
        Y (np.ndarray): The target universe matrix of shape (n, 300).
        R (np.ndarray): The initial transformation matrix of shape (300, 300).
        train_steps (int): Number of Gradient Descent iterations.
        learning_rate (float): Step size for Gradient Descent update.
        
    Returns:
        np.ndarray: The optimized transformation matrix R.
    """
    n = X.shape[0]
    
    for i in range(train_steps):
        # Forward pass: predictions of X in the new space
        predictions = np.dot(X, R)
        
        # Calculate the gradient of the Frobenius norm loss function
        # dJ/dR = (2/n) * X^T (XR - Y)
        error = predictions - Y
        gradient = (2.0 / n) * np.dot(X.T, error)
        
        # Update transformation matrix R using Gradient Descent
        R -= learning_rate * gradient
        
    return R

def nearest_neighbor(v, candidates, k=1):
    """
    Find the k nearest neighbors in the candidates matrix to the vector v 
    using Cosine Similarity.
    
    Args:
        v (np.ndarray): The target vector of shape (300,).
        candidates (np.ndarray): The matrix of candidate vectors of shape (m, 300).
        k (int): Number of nearest neighbors to retrieve.
        
    Returns:
        list: Indices of the top k most similar vectors in candidates.
    """
    # Calculate dot products between candidates and v
    dot_products = np.dot(candidates, v)
    
    # Calculate L2 norms (Euclidean magnitude) using basic numpy operations
    # ||A|| = sqrt(sum(A_i^2))
    norm_v = np.sqrt(np.sum(v ** 2))
    norms_candidates = np.sqrt(np.sum(candidates ** 2, axis=1))
    
    # Calculate cosine similarity: (A . B) / (||A|| * ||B||)
    # Added a small epsilon (1e-8) to avoid division by zero
    similarity_l = dot_products / (norms_candidates * norm_v + 1e-8)
    
    # argsort sorts in ascending order, so the highest similarities are at the end
    sorted_ids = np.argsort(similarity_l)
    
    # Return the indices of the top k highest similarities
    return sorted_ids[-k:].tolist()
