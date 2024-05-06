import argparse
from PattRecClasses import GaussD
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description='HMM for action recognization')
    parser.addargument('iA', type=np.array, 
                       default=np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]), help='initial A')
    parser.addargument('q', type=np.array, 
                       default=np.array([1, 0, 0]), help='initial state matrix')
    parser.addargument('g1', type=GaussD, 
                       default=GaussD(means=[0, 0, 0], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), help='parameters of G1')
    parser.addargument('g2', type=GaussD, 
                       default=GaussD(means=[0, 0, 0], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), help='parameters of G2')
    parser.addargument('g3', type=GaussD, 
                       default=GaussD(means=[0, 0, 0], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), help='parameters of G3')


    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train (default: 110)')
    
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, specify 1,3 for example.')


    args = parser.parse_args()

    return args