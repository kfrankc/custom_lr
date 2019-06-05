import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import copy as cp
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# logging library
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.propagate = True

def computecost(beta, lam, X, y):
    '''
    Computes and returns F(β) for any β. Note: Avoid using for loops by vectorizing the computation
    Input:
        beta  = d x 1 matrix
        l     = lambda
        X     = n x d matrix
        y     = n x 1 matrix
    Output: 
        cost  = F(beta)
    '''
    n = len(y)
    tmp = np.zeros([X.shape[0], 1])
    for i in range(n):
        y_i = y[i].reshape(-1,1)
        X_i = X[i].reshape(-1,1)
        score = y_i.dot(X_i.T).dot(beta)
        tmp[i] = np.log(1 + np.exp(-score))
    cost = (1/n) * np.sum(tmp) + lam*(beta.T.dot(beta))
    return cost

def computegrad(beta, lam, X, y):
    '''
    Computes and returns ∇F(β) for any β.
    Input:
        beta  = d x 1 matrix
        l     = lambda
        X     = n x d matrix
        y     = n x 1 matrix
    Output: 
        grad  = gradient of F(beta)
    '''
    n = len(y)
    P = np.zeros([X.shape[0], 1])
    for i in range(n):
        y_i = y[i].reshape(-1,1)
        X_i = X[i].reshape(-1,1)
        score = y_i.dot(X_i.T).dot(beta)
        sigmoid = 1/(1 + np.exp(-score))
        P[i] = np.exp(-score) * sigmoid
    P = np.diagflat(P)
    grad = -(1/n)*X.T.dot(P).dot(y) + 2*lam*beta
    return grad

def backtracking(beta, step_size, lam, X, y):
    '''
    Computes and returns eta_t for any β and theta.
    Input:
        beta      = d x 1 matrix
        step_size = eta value
        l         = lambda
        X         = n x d matrix
        y         = n x 1 matrix
    Output: 
        eta   = new step size
    '''
    tmp = beta - step_size * computegrad(beta, lam, X, y)
    left = computecost(tmp, lam, X, y)
    grad_beta = computegrad(beta, lam, X, y)
    right = computecost(beta, lam, X, y) - (step_size/2) * grad_beta.T.dot(grad_beta)
    while (left > right):
        logger.info("backtracking... left: {}, right: {}".format(left, right))
        step_size = step_size * 0.8
    return step_size

def fastgradalgo(beta, theta, step_size, e, lam, X, y, max_iter, print_iter, log):
    '''
    Implements the fast gradient descent algorithm described in Algorithm 1. 
    The function graddescent calls the function computegrad and backtracking as a sub-routine. 
    Input:
        beta         = d x 1 matrix
        theta        = d x 1 matrix for fast gradient 
        step_size    = the eta value
        e            = target accuracy
        lam          = lambda
        X            = n x d matrix
        y            = n x 1 matrix
    Output:
        beta         = beta_best
        cost_history = array of cost to be plotted later
    '''
    n = len(y)
    cost_history = []
    beta_history = []
    total_iterations = 1
    t = 0
    while (t < max_iter and np.linalg.norm(computegrad(beta, lam, X, y)) > e):
        if (total_iterations % print_iter == 0 and log == True):
            logger.info("FG - iteration: {}".format(total_iterations))
            logger.info("FG - norm(gradient): {}".format(np.linalg.norm(computegrad(beta, lam, X, y))))
        cost = computecost(beta, lam, X, y)
        cost_history.append(cost[0][0])
        beta_history.append(beta)
        step_size = backtracking(beta, step_size, lam, X, y)
        if (total_iterations % print_iter == 0 and log == True):
            logger.info("FG - cost: {}".format(cost[0][0]))
            logger.info("FG - new step_size: {}".format(step_size))
        beta_old = cp.copy(beta)
        beta = theta - step_size*computegrad(theta, lam, X, y)
        theta = beta + (t / (t + 3)) * (beta - beta_old)
        total_iterations += 1
        t += 1
    # convert to np.array
    cost_history = np.asarray(cost_history)
    beta_history = np.asarray(beta_history)
    return beta, cost_history, beta_history, total_iterations

class LogisticRegression():
    def __init__(self, lam=1.0, max_iter=1000, print_iter=100, e=1e-3, log=True):
        self.lam_ = lam
        self.max_iter_ = max_iter
        self.print_iter_ = print_iter
        self.e_ = e
        self.log_ = log

    def fit(self, X, y):
        eigenValues, eigenVectors = np.linalg.eigh((1/len(y)) * X.T.dot(X))
        L = max(eigenValues) + self.lam_
        step_size = 1/L
        if (self.log_ == True):
            logger.info("initial step_size: {}".format(step_size))
        # Initialize beta, theta to 0
        beta = np.zeros([X.shape[1], 1])
        theta = np.zeros([X.shape[1], 1])
        # Run gradient descent
        beta_best_fg, cost_history_fg, beta_history_fg, total_iterations_fg = \
            fastgradalgo(beta, theta, step_size, self.e_, self.lam_, X, y, self.max_iter_, self.print_iter_, self.log_)
        self.coef_ = beta_best_fg
        self.cost_history_ = cost_history_fg
        self.beta_history_ = beta_history_fg
        self.total_iterations_ = total_iterations_fg
        return self

    def visualize(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_ylabel('F(β)', fontsize=16)
        ax.set_xlabel('Number of Iterations', fontsize=16)
        plt.title("F(β) vs. Iterations", fontsize=18)  
        plt.plot(range(self.total_iterations_-1), self.cost_history_)
        plt.pause(0.001)
        input("Press [enter] to continue.")
        fig.savefig('demo_fastgrad.png')


    def score(self, X, y):
        y_pred = expit(np.dot(X, self.coef_))
        y_pred[y_pred < 0.5] = -1
        y_pred[y_pred >= 0.5] = 1
        result = accuracy_score(y, y_pred)
        return result
