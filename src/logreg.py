"""Logistic regression built from scratch with Newton's method.

Model: chance of default p = 1 / (1 + exp(-(A @ theta)))
Loss : average log-loss (optionally weighted) plus a small L2 penalty
"""
import numpy as np
from scipy.special import expit


def objective(theta, A, y, sw, pen):
    z = A @ theta
    return np.mean(sw * (np.logaddexp(0, z) - y * z)) + 0.5 * np.sum(pen * theta ** 2)


def gradient(theta, A, y, sw, pen):
    # the formula we derived: (p - y) times x, averaged, plus the penalty's slope
    p = expit(A @ theta)
    return A.T @ (sw * (p - y)) / len(y) + pen * theta


def hessian(theta, A, sw, pen):
    # how the slope itself bends: average of p(1-p) times x times x-transpose
    p = expit(A @ theta)
    W = sw * p * (1 - p)
    return (A * W[:, None]).T @ A / len(A) + np.diag(pen)


def fit_newton(A, y, sample_weight=None, penalty=None, theta0=None,
               max_iter=50, tol=1e-9):
    """Returns (theta, number of Newton steps taken)."""
    n, d = A.shape
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    pen = np.zeros(d) if penalty is None else np.asarray(penalty, float)
    theta = np.zeros(d) if theta0 is None else np.array(theta0, float)
    steps = 0
    while steps < max_iter:
        g = gradient(theta, A, y, sw, pen)
        if np.linalg.norm(g) < tol:
            break
        theta = theta - np.linalg.solve(hessian(theta, A, sw, pen), g)
        steps += 1
    return theta, steps


def predict_proba(A, theta):
    return expit(A @ theta)
