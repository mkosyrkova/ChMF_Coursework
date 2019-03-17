from math import pi
from math import sin
from math import cos
from math import exp
from math import pow
import numpy
import matplotlib.pyplot as plt

#инициализация констант
T = 15
I = 100 # узнать чему равно I
R = 6
K = 100 # узнать чему равно K
l = 2*pi*R
hx = l/I
ht = T/K
i = numpy.arange(0, I+1)
k = numpy.arange(0, K+1)
xi = i*hx
tk = k*ht
gamma = ht*0.065/(1.84*hx*hx)

U = numpy.zeros((K+1,I+1), dtype=float)

# инициализация начальной функции psi
# и задание начального условия
psi = numpy.zeros(I+1)
for i_iter in range(0,I):
    if xi[i_iter] < l/2:
        psi[i_iter] = 1
    U[0, i_iter] = psi[i_iter]

for k_iter in range(0,K-1):
        U[k_iter+1, 0] = U[k_iter,0]*(1 - 2*gamma) + gamma*(U[k_iter, 1]+U[k_iter, I-1])
        U[k_iter+1, I] = U[k_iter+1, 0]
        for i_iter in range(1,I-1):
            U[k_iter+1, i_iter] = U[k_iter,i_iter]*(1 - 2*gamma) + gamma*(U[k_iter, i_iter+1]+U[k_iter, i_iter-1])

plt.plot(xi,U[K-1,])
plt.show()

