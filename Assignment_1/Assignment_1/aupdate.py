import numpy as np
from PattRecClasses import GaussD

def aupdate(A, alpha, beta, c, bmat):
    yp = np.zeros(len(alpha),A.shape[0], A.shape[1])
    ypb = np.zeros(A.shape[0], A.shape[1])
    al = alpha.T
    be = beta.T

    for t in range(len(alpha)-1):
        yp[t,:,:] = al[:,t]@A[:,:]@(bmat[:,t+1]*be[:,t+1])

    ypb = np.sum(yp, axis=0)

    A_update = np.zeros(A.shape[0], A.shape[1])
    for i in range(A.shape[0]):
        A_update[i,:] = ypb[i,:]/np.sum(ypb[i,:])

    return A_update

def bupdate(gamma, g, data):
    mean = np.zeros(len(g))
    gam = gamma.T
    for i in range(len(g)):
        mean[i]= data[:,:]*gam[i,:]
        mean[i] = np.sum(mean[i])/np.sum(gam[i,:])
    
    varn = np.zeros(len(g))
    data1 = data-mean
    data1 = data1**2
    data1 = gam@data1
    data1 = np.sum(data1, axis=0)
    varn = data1/np.sum(gam,axis=1)

    return mean, varn

       