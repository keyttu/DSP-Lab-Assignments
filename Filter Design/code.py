import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# Define Desired Frequency Response
w = np.linspace(-np.pi, np.pi, 500)
Hd_w = np.zeros(len(w))

def desResp(a):
  if(a>-0.4*np.pi and a<0.4*np.pi):
    return 1
  else:
    return 0

for i in range(len(w)):
  Hd_w[i] = desResp(w[i])

plt.stem(w/np.pi, Hd_w)
plt.xlabel('Normalized Frequency')
plt.ylabel('Amplitude')
plt.show()

def ifft(X,w):
  l = len(X)
  n = np.arange(-100,100)
  x = np.zeros(len(n))
  for i in range(len(n)):
    ans = 0
    for j in range(len(X)):
      ans+=X[j]*np.exp(1j*2*np.pi*n[i]*w[j])
    x[i] = ans
  return x

x = ifft(X,w)
plt.plot(x)



~~~~~~~~~~~~~

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def fft(X,w):
  l = len(X)
  n = np.linspace(-np.pi,np.pi,2000)
  x = np.zeros(len(n))
  for i in range(len(n)):
    ans = 0
    for j in range(len(X)):
      ans+=X[j]*np.exp(1j*2*np.pi*n[i]*w[j])
    x[i] = ans
  return x

key = np.linspace(-10,10,1000)
x = np.zeros(len(key))
for i in range(len(x)):
  if(key[i]!=0):
    x[i] = np.sin(key[i]*np.pi)/(key[i]*np.pi)
  else:
    x[i] = 1

plt.plot(x)
plt.show()
X = fft(x,key)

n = np.linspace(-np.pi,np.pi,2000)
plt.plot(n/np.pi,20*np.log10(abs(X)))
plt.show()
