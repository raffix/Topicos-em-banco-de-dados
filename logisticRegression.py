import random
import numpy as np
import sys
import matplotlib.pyplot as pl

def J(X, y, th):
  m = X.shape[0]
  y_hat = g(X.dot(th.T))
  l1 = y * np.log(y_hat)
  l2 = (1 -y) * np.log(y_hat)
  error = (l1 + l2).sum()
  return -1 * (error/m)

def GD (X, y, th, alpha, niters):
  m=X.shape[0]
  cost = np.zeros((niters,1))
  for i in range(0,niters):
    y_hat = X.dot(th.T)
    erro = ((y_hat-y)*X).sum(0)/m
    th = th-(alpha*(error))
    cost[i]=J(X,y,T)
  return T,cost
  
 def g(X, th):
  return 1/ (1+ (np.e**(-(X.dot(th.T))))