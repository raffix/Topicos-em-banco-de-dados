import numpy as np
import matplotlib.pyplot as ply
import sys

alpha = 0.01

def custo(X, Y, theta):
	m = 2 * X.shape[0]
	yHat = X.dot(theta.T)
	error = ((yHat-Y)**2).sum()
	return error/(m)

def gradiente(x, y, theta, alpha, nitens):
	m = x.shape[0]
	costs = np.zeros((nitens,1))
	for i in range(0, nitens):
		yHat = x.dot(theta.T)
		error = (yHat - y) * x
		error = error.sum(0)/m
		theta = theta - (alpha*error)
		costs[i] = custo(x, y, theta)
	return theta, costs

def main():
	f = open('arquivo.txt', 'r')
	rows = [l.split(',') for l in f.readlines()]
	m = len(rows)
	d = len(rows[0])-1 ## n labels
	y = np.array([l[d:d+1] for l in rows], dtype = float)
	x = np.array([l[0:d] for l in rows], dtype= float)
	theta = np.zeros((1, d+1))
	x = np.insert(x, 0, 1, axis=1)
	theta,j = gradiente(x, y, theta, 0.01, 1000)
	ply.plot(x[:,1:], y,'*')
	yHat = x.dot(theta.T)
	ply.plot(x[:,1:], yHat,'-')
	ply.show()
	
	'''
	ply.plot(xtrain[,1:], ytrain, '*', c='S')
	yHat = xtrain.dot(theta.T)
	ply.plot(xtrain[:,1:], yHat, '-', c= 'r')
	ply.show()
	
	ply.plot(xtest[:,1:], ytest, '*', c='b')
	yHat = xtest.dot(theta.T)
	ply.plot(xtest[:,1:], yHat, '-', c='r')
	ply.show()
	'''

main()
