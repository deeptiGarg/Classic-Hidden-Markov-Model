
# coding: utf-8

# In[180]:


import numpy as np
import random
import math


# In[203]:


class HMM:
    def __init__(self,n,seed,o):
        self.N = n
        self.M = 27
        self.T = len(o)
        self.A = np.zeros((self.N,self.N))
        self.B = np.zeros((self.N,self.M))
        self.pi = np.zeros((1,self.N))
        self.initialize(seed)
        self.minIters = 100
        self.oldlogProb = -1.0
#         New LogProb 
        self.logProb = 0.0
        self.α = np.zeros((self.T,self.N),dtype=np.float64)
        self.β = np.zeros((self.T,self.N),dtype=np.float64)
        self.ϒ = np.zeros((self.T,self.N))
        self.diGamma = np.zeros((self.T,self.N,self.N))
        self.c = np.zeros((1,self.T))
        
# Initialize A[][], B[][] and pi[]    
    def initialize(self,seed):
        random.seed(seed)        
        # Initialize A        
        prob = 1.0 / self.N;
        for i in range(0,self.N):
            sumN = 0.0;
            for j in range(0,self.N):
                if(random.randint(1,100)%2 == 0):
                    self.A[i][j] = prob + float(random.randint(1,3999)*prob)/100000
                else:
                    self.A[i][j] = prob - float(random.randint(1,3999)*prob)/100000
                sumN = sumN + self.A[i][j];
            for j in range(0,self.N):
                self.A[i][j] = self.A[i][j]/sumN;
        print("A=",self.A)
        
        # Initialize B       
        prob = 1.0 / self.M;
        for i in range(0,self.N):
            sumM = 0.0;
            for j in range(0,self.M):
                if(random.randint(1,100)%2 == 0):
                    self.B[i][j] = prob + float(random.randint(1,3999)*prob)/100000
                else:
                    self.B[i][j] = prob - float(random.randint(1,3999)*prob)/100000  
                sumM = sumM + self.B[i][j];
            for j in range(0,self.M):
                self.B[i][j] = self.B[i][j]/sumM;
        
        # Initialize pi       
        prob = 1.0 / self.N;
        sumN = 0.0;
        for i in range(0,self.N):    
            if(random.randint(1,100)%2 == 0):
                self.pi[0][i] = prob + float(random.randint(1,3999)*prob)/100000
            else:
                self.pi[0][i] = prob - float(random.randint(1,3999)*prob)/100000
            sumN = sumN + self.pi[0][i];
        for i in range(0,self.N):
            self.pi[0][i] = self.pi[0][i]/sumN;
        print("B=",self.B)
        print("Pi=",self.pi)
    
    def alphaPass(self,o):
        #α = np.zeros((self.T,self.N),dtype=np.float64)
        #c = []
        # Compute α0(i)
        self.c[0][0] = 0
        for i in range(0,self.N):
            self.α[0][i] = self.pi[0][i]*self.B[i][o[0]]
            self.c[0][0] = self.c[0][0] + self.α[0][i]
        
        # Scale the α0(i)
        self.c[0][0] = 1/self.c[0][0]
        for i in range(0,self.N):
             self.α[0][i] = self.c[0][0]* self.α[0][i]
        
        # Compute αt(i)
        for t in range(1,self.T):
            self.c[0][t] = 0
            for i in range(0,self.N):
                self.α[t][i] = 0
                for j in range(0,self.N):
                    self.α[t][i]  = self.α[t][i] + self.α[t-1][j]*self.A[j][i]
                self.α[t][i] = self.α[t][i]*self.B[i][o[t]]
                self.c[0][t] = self.c[0][t] + self.α[t][i]
                #  Scale αt(i)
            print("self.α=",self.α)
            self.c[0][t]= 1/self.c[0][t]
            for i in range(0,self.N):
                self.α[t][i] = self.c[0][t]* self.α[t][i]
            
    def betaPass(self,o):
        #β = np.zeros((self.T,self.N),dtype=np.float64)
        
        for i in range(0,self.N):
            self.β[self.T-1][i] = self.c[0][self.T-1]
        
        # β-pass
        for t in range(self.T-2, -1, -1):
            for i in range(0,self.N):
                self.β[t][i] = 0
                for j in range(0,self.N):
                    self.β[t][i]  = self.β[t][i] + self.A[i][j]*self.B[j][o[t+1]]*self.β[t+1][j]
                
                # Scale β[t][i] with same scale factor as α[t][i]
                self.β[t][i] = self.c[0][t]*self.β[t][i]
    
    def gamma(self,o):
        #ϒ = np.zeros((self.T,self.N))
        #diGamma = np.zeros((self.T,self.N,self.N))
        for t in range(0,self.T-1):
            denom = 0
            for i in range(0,self.N):
                for j in range(0,self.N):
                    denom = denom + self.α[t][i]*self.A[i][j]*self.B[j][o[t+1]]*self.β[t+1][j]
            for i in range(0,self.N):
                self.ϒ[t][i] = 0
                for j in range(0,self.N):
                    self.diGamma[t][i][j] = (self.α[t][i]*self.A[i][j]*self.B[j][o[t+1]]*self.β[t+1][j])/denom
                    self.ϒ[t][i] = self.ϒ[t][i] + self.diGamma[t][i][j]
        
        # Special case for ϒ[T-1][i]
        denom = 0
        for i in range(0,self.N):
            denom = denom + self.α[self.T-1][i]
        for i in range(0,self.N):
            self.ϒ[self.T-1][i] = self.α[self.T-1][i]/denom
    
    def reEstimate(self,o):
        # Re-estimate pi
        for i in range(0,self.N):
            self.pi[0][i] = self.ϒ[0][i]

        # Re-estimate A
        for i in range(0,self.N):
            for j in range(0,self.N):
                numer = 0
                denom = 0
                for t in range(0,self.T-1):
                    numer = numer + self.diGamma[t][i][j]
                    denom = denom + self.ϒ[t][i]
                self.A[i][j] = numer/denom

        # Re-estimate B
        for i in range(0,self.N):
            for j in range(0,self.M):
                numer = 0
                denom = 0
                for t in range(0,self.T):
                    if(o[t]==j):
                        numer = numer + self.ϒ[t][i]
                    denom = denom + self.ϒ[t][i]
                self.B[i][j] = numer/denom

    def computeLog(self):
        logProb = 0.0
        for i in range(0,self.T):
            logProb = logProb + math.log(self.c[0][i])
        self.logProb = -logProb
        
    def control(self,o):
        iters = 0
        while(iters<self.minIters and (self.logProb > self.oldlogProb)):
            self.oldlogProb = self.logProb;
            # run once for first iter
            self.alphaPass(o)
            self.betaPass(o)
            self.gamma(o)
            self.reEstimate(o)
            self.computeLog()
            if(iters == 0):
                self.oldlogProb = self.logProb - 1.0
            iters = iters + 1
            print("Iteration Completed= {0}, log [P(observation | lambda)] ={1}".format(iters, self.logProb))
        print("Total iterations =",iters)
        print("log [P(observations | lambda)] =", self.logProb)
        print("Final pi =",self.pi)
        print("Final A =",self.A)


# In[196]:


def preProcess():
    text = ""
    with open('brownShort.txt', 'r') as myfile:
        text=myfile.read().replace('\n', '')
    newTextO = list()
    #text = "abis 89a 7Arc -is z"
    i = 0
    for char in text:
        if(ord(char.lower())>=97 and ord(char.lower())<=122):
            newTextO.append(ord(char.lower())%97)
            i = i+1
        else:
            if(ord(char.lower())==32):
                newTextO.append(26)
                i = i+1
        if(i==50000):
            break
    return newTextO


# In[197]:


obSeq = preProcess()


# In[199]:




# In[204]:


p = HMM(2,2,obSeq)
p.control(obSeq)


# In[110]:

