# Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# Design of FIR Low pass filter
def sinc(Fc,T):
    n = np.linspace(-5,5,T)
    x = np.zeros(len(n))
    for i in range(len(x)):
      if(n[i]!=0):
        x[i] = np.sin(n[i]*np.pi*2*Fc)/(n[i]*np.pi*2*Fc)
      else:
        x[i] = 1
    return x,n

def fft(X,w):

  n = np.linspace(0,np.pi,500)
  x = np.zeros(len(n))
  for i in range(len(n)):
    ans = 0
    for j in range(len(X)):
      ans+=X[j]*np.exp(1j*2*np.pi*n[i]*w[j])
    x[i] = ans
  return x,n

def FILTERDesign(Filter, fc1, fc2, window,T):
    x,n = sinc(fc1,T)
    win = np.ones(len(n))
    if(window == "hamming"):
        win = sig.hamming(len(n))
    if(window == "hanning"):
        win = sig.hanning(len(n))
    if(window == "blackman"):
        win = sig.blackman(len(n))
    if(window == "triangular"):
        win = sig.triang(len(n))
        
    if(Filter == "LPF"):
        # Low pass filter
        x,n = sinc(fc1,T)
        X,w = fft(x*win,n)
        return X/max(X),w
    if(Filter == "HPF"):
        # High Pass Filter
        x,n = sinc(np.pi-fc1,T)
        X,w = fft(x*win,n)
        X = X/max(X)
        return np.flip(X),w
    if(Filter == "BPF"):
        # Band Pass filter
        f1 = fc1
        f2 = fc2
        
        if(f1<f2):
            x1,n1 = sinc(np.pi-f1,T)
            x2,n2 = sinc(f2,T)
            X1,w = fft(x1*win,n1)
            X2,w = fft(x2*win,n2)
            X1 = X1/max(X1)
            X2 = X2/max(X2)
            return (np.flip(X1)*X2)/max(np.flip(X1)*X2),w
        
    if(Filter == "BRF"):
        # Band Pass filter
        f1 = fc1
        f2 = fc2
        
        if(f1<f2):
            x1,n1 = sinc(f1,T)
            x2,n2 = sinc(np.pi-f2,T)
            X1,w = fft(x1*win,n1)
            X2,w = fft(x2*win,n2)
            X1 = X1/max(X1)
            X2 = X2/max(X2)
      
            return (np.flip(X2)+(X1))/max(np.flip(X2)+X1),w


fil1,w = FILTERDesign("LPF",0.3*np.pi,0.6*np.pi,'hanning',100)
fil2,w = FILTERDesign("HPF",0.3*np.pi,0.6*np.pi,'',100)
fil3,w = FILTERDesign("BPF",0.3*np.pi,0.6*np.pi,'',100)
fil4,w = FILTERDesign("BRF",0.2*np.pi,0.8*np.pi,'hanning',100)

