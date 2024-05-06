from PattRecClasses import GaussD, HMM, MarkovChain
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from tqdm import tqdm
from parse_args_train import parse_args
import pandas as pd
from aupdate import aupdate, bupdate

class Trainer(object):
     def __init__(self, args):
        self.args = args
        g = [args.g1, args.g2, args.g3]
        self.g = g
        df = pd.read_csv('train.csv')
        usefuldata = df.iloc[:, 1:4]
        self.data = usefuldata.values

        df = pd.read_csv('test.csv')
        usefuldata1 = df.iloc[:, 1:4]
        self.tdata = usefuldata1.values

     def training(self):
         mc = MarkovChain(self.args.q, self.args.iA)
         HMM1  = HMM(mc, self.g) 
         bmat = HMM1.prob(self.data, True)

         alpha, c = mc.forward(bmat)
         beta = mc.backward(bmat, c)

        #compute gamma
         gamma = np.empty(len(alpha))
         for i in range(len(alpha)):
             gamma[i] = alpha[i,:]*beta[i,:]*c[i]
            
        #update q
         q_update = gamma[0,:]

        #update A
         A_update = aupdate(self.args.iA, alpha, beta, c, bmat)

        #update B
         mean, varn = bupdate(gamma, self.g, self.data)
    
         return q_update, A_update, mean, varn
     

     def testing(self):
        mc = MarkovChain(self.args.q, self.args.iA)
        HMM1  = HMM(mc, self.g) 
        bmat = HMM1.prob(self.tdata, True)
    
        alpha, c = mc.forward(bmat)
        beta = mc.backward(bmat, c)
    
        #compute gamma
        gamma = np.empty(alpha.shape)
        for i in range(alpha.shape[0]):
            gamma[i,:] = (alpha[i,:]*beta[i,:])*c
        staten = np.zeros(alpha.shape[1]) 
        staten = np.argmax(gamma, axis=0)

        return staten
        
     
def main(args):
    trainer = Trainer(args)
    for i in range(args.epochs):
        q_update, A_update, mean, varn = trainer.training()
        args.q = q_update
        args.iA = A_update
        args.g1 = GaussD(means=mean[0], cov=varn[0])
        args.g2 = GaussD(means=mean[1], cov=varn[1])
        args.g3 = GaussD(means=mean[2], cov=varn[2])

    print('q:', args.q)
    print('A:', args.iA)
    print('g1:', mean[0], varn[0])
    print('g2:', mean[1], varn[1])
    print('g3:', mean[2], varn[2])

    staten = trainer.testing()
    plt.plot(staten)
    plt.xlabel("time axis")
    plt.ylabel("states")
    plt.title("testing results")


    # correct = 0
    # for i in range(len(staten)):
    #     if staten[i]==label[i]:
    #         correct += 1
    # print('accuracy:', correct/len(staten))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('1')

         

     
         
    


