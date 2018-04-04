import random
import numpy as np
import sys
import matplotlib.pyplot as pl

def RMSE(y, y_hat, m):
	return np.sqrt((1/m) * (np.sum(y_hat - y) ** 2))

## returns the predicted y (based on feature dataset (X) and weights (Theta))
def F(X,T):
    return X.dot(T.T)

## returns the cost (residual error) for predicted y
def J(X,y,T):
  m=X.shape[0]
  y_hat=X.dot(T.T)
  cost=np.sum((y_hat-y)**2)/(2*m)
  return cost

## finds the "good" values for the weigths (Theta)
## Gradient descent (optmization function)
def GD (X,y,T,alpha,niters):
  m=X.shape[0]
  cost=np.zeros((niters,1))
  for i in range(0,niters):
    y_hat=X.dot(T.T)
    error=((y_hat-y)*X).sum(0)/m
    T=T-(alpha*(error))
    cost[i]=J(X,y,T)
  return T,cost
  
def normalizeDataset(X, m, d):
	return (X - np.min(X,0)) / (np.max(X, 0) - np.min(X,0))

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print('Usage: python %s <dataset file name (full path)>' % sys.argv[0])
        exit(0)

    try:
        f=open(sys.argv[1]) ## open the dataset file
    except:
        print('Could not open',sys.argv[1])
        exit(0)
    ## rows will be a list of list, e.g., [[example 1],[example 2],...[example m]]
    rows=[l.split(',') for l in f.readlines()] ## the values are separated by comma
    m=len(rows) ## how many lists rows has, i.e., how many examples in the dataset
    ## how many features are in a given list
    d=len(rows[0])-1 ## we subtract the result from 1 to discard the label (y)
    y=np.array([l[d:d+1] for l in rows],dtype=float) ## vector of dataset labels
    X=np.array([l[0:d-1] for l in rows],dtype=float) ## matrix of dataset features
    Theta=np.zeros((1,d)) ## Initialize Theta with zeros
    
    X = normalizeDataset(X, m, d)
    
    X=np.insert(X,0,1,axis=1) ## inserts a column of 1's
    tsize=int(m*0.7) ## size of the training set
    Xtr=X[:tsize,:] ## from the first row to tsize-1, all colmuns
    Xte=X[tsize:,:] ## from the tsize row to the end, all columns
    ytr=y[:tsize]
    yte=y[tsize:]
    ### Call gradient descent to find the "good" values to Theta
    Theta,cost=GD(Xtr,ytr,Theta,0.01,3000)
    print("Test cost: ", J(Xte, yte, Theta))
    pl.plot(cost)
    pl.show()
    y_hat=F(Xte,Theta)
    
    print(RMSE(yte, yte.dot(Theta), m))
    
    for i in range(10):
	    print(y_hat[i], '<-->', yte[i], '=>', np.abs(y_hat[i] - yte[i]))
    if d>1:
        print('Sorry, we cannot plot more than 1 feature')
        exit(0)
    ## predict new labels from training set
    y_hat=Xtr.dot(Theta.T)
    ## Plot the predict y's in relation to the ground-truth (y)
    ## first the X regarding the ground-truth
    pl.plot(Xtr[:,1:],ytr,'.',c='b')
    ## now X regarding the predict values
    pl.plot(Xtr[:,1:],y_hat,'-',c='g')
    pl.show()
    ## Now, let's check the performance on the test set
    ## y_hat is built from Xte, i.e., the test set
    y_hat=Xte.dot(Theta.T)
    pl.plot(Xte[:,1:],yte,'.',c='b')
    ## now X regarding the predict values
    pl.plot(Xte[:,1:],y_hat,'-',c='g')
    pl.show()
    print("Test cost", J(Xte,yte,Theta))
    pos = (y==1).ravel()
    neg = (y==0).ravel()
    pl.plot(X[pos,1], X[pos, 2], 'o', c='r')
    pl.plot(X[neg,1], X[neg, 2], 'o', c='b')
