'''
SNU Big data academy
"GMM estimation with EM iteration"

'''


import numpy as np
import os
import scipy
import time
import math
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def N(data, mu, sigma):
    x = data
    prob = np.exp( -0.5*np.matmul(np.matmul(np.transpose(x-mu[...,None]),np.linalg.inv(sigma)),x-mu[...,None]) )/ np.sqrt(np.linalg.det(2*math.pi*sigma))
    return prob


# Data generation
np.random.seed(1234)
mean1 = [0, 0]
cov1 = [[1, 0], [2, 12]]
x1, y1 = np.random.multivariate_normal(mean1,cov1,300).T

np.random.seed(1234)
mean2 = [5, 0]
cov2 = [[17, 2], [1.3, 2]]
x2, y2 = np.random.multivariate_normal(mean2,cov2,300).T

x = np.hstack((x1,x2))
y = np.hstack((y1,y2))
data = np.vstack((x,y))

# Plot : labeled data points
plt.ion()
plt.figure()
plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'ro')
plt.title('Ground truth')
plt.show()
plt.pause(0.5)
input('Press <Enter> to continue')

# Plot : unlabeled data points
plt.figure()
plt.plot(x,y,'bo')
plt.title('Data to be clustered')
plt.show()
plt.pause(0.5)
input('Press <Enter> to continue')

# GMM algorithm
nCluster = 2
np.random.seed(8888)

mu = np.random.random((2,nCluster)) # -> mean of the gaussian
sigma = np.random.random((2,2,nCluster))+np.repeat(np.identity(2)[...,None],nCluster,axis=2)    # -> std of the gaussian
pi = np.ones(nCluster)/nCluster

posterior = np.zeros((nCluster,len(x))) # -> posterior probability (mixture | data, parameter)

X, Y = np.meshgrid(np.linspace(-7,16,56), np.linspace(-10,10,40))
Z = np.zeros((X.shape[0],X.shape[1],nCluster))
plt.figure()
# EM
for i in range(0,30):
    # E-step
    for j in range(0,nCluster):
        for k in range(0,len(x)):
            posterior[j,k] = pi[j]*N(data[:,k:k+1],mu[:,j],sigma[:,:,j])
    S = np.sum(posterior, axis=0)
    posterior = posterior/S

    # M-step
    for j in range(0,nCluster):
        mu[:,j] = np.sum(posterior[j,:]*data,axis=1)/np.sum(posterior[j,:])

    for j in range(0,nCluster):
        sigma[:,:,j] = 0
        for k in range(0,len(x)):
            sigma[:,:,j] += (np.matmul(  (posterior[j,k]*(data[:,k]-mu[:,j]))[...,None] ,np.transpose((data[:,k]-mu[:,j])[...,None]) ))/np.sum(posterior[j,:])

    for j in range(0,nCluster):
        pi[j] = np.sum(posterior[j,:])/len(x)


    for j in range(0,nCluster):
        Z[:,:,j] = np.diagonal(N( np.vstack((X.flatten(),Y.flatten())),mu[:,j],sigma[:,:,j])).reshape(X.shape[0],X.shape[1])

    # Plot for real-time visualization
    plt.clf()
    plt.contour(X, Y, Z[:,:,0]) # GMM contour
    plt.contour(X, Y, Z[:,:,1])

    idx = (np.argmax(posterior,axis=0)==0)  # 1st mixture
    plt.plot(data[0,idx],data[1,idx],'bo')
    idx = (np.argmax(posterior,axis=0)==1)  # 2nd mixture
    plt.plot(data[0,idx],data[1,idx],'ro')
    plt.plot(mu[0,:],mu[1,:],'kX', markerSize=20)   # center of Gaussian distribution
    plt.title('GMM estimation with EM algorithm (iteration='+str(i+1)+')')
    plt.show()
    plt.pause(0.5)
    #input('Press <Enter> to continue')
