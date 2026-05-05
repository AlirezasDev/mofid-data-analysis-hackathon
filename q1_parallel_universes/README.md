# Question 1: Parallel Universes Transformation

This project finds an optimal transformation matrix between two parallel universes
using gradient descent optimization and identifies nearest neighbors via cosine similarity.

## Problem

Given two universe representations as 300-dimensional vectors:
- Source universe: matrix X (shape: n × 300)
- Target universe: matrix Y (shape: n × 300)

Find a transformation matrix R (shape: 300 × 300) that minimizes:

$$J = \frac{1}{n} ||Y - XR||_F^2$$

Where $||·||_F$ is the Frobenius norm.

## Solution

- `train_transformation`: Uses gradient descent to optimize R
- `nearest_neighbor`: Finds k nearest neighbors using cosine similarity

## Implementation

Uses only NumPy for:
- Vectorized gradient computation
- Efficient matrix operations
- Cosine similarity calculations

## Files

- `submit.py`: Contains the required functions
- `X_train.npy`: Source universe matrix
- `Y_train.npy`: Target universe matrix
