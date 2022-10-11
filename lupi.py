import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt

"""
Lo scopo è ricavare il numero di lupi di una foresta sapendo:
che in una pirma misurazione ne sono stati visti e targati: M
in un secondo momento si sono visti altri lupi di numero : n
di questi n lupi r erano già stati visti all'epoca di M,
cosa si può dire su N?
Per il teorema di Bayes:
P(N|Mnr) = (P(r|NMn)*P(N|Mn))/P(r|Mn)

P(r|NMn) => ipergeometrica

P(N|Mn)  => come dipende il numero di lupi da M ed n?
            ne abbiamo visti prima M e poi n quini
            almeno N_min sara il massiomo tra i due.
            Sarebbe la probabilità a priori

P(r|Mn)  => Normalizzazione calcolata a mano
"""

def prob_priori(Nmin, Nmax):
    """
    Probabilita a priori distribuzione lupi
    """
    return 1/(Nmax - Nmin)

M = 20  #numero lupi visti all'inizio
n = 30  #numero di lupi osservati alla seceonda volta

Nmin = max(M, n)
Nmax = 300
N = np.arange(Nmin, Nmax) #intevallo su cui variare il numero di lupi
r = 7 # il numero di lupi osservati che erano gia stati visti la prima volta

#array ceh conterrà la nostra curva
A = np.array([])

for nn in N: #ciclo sul numero di lupi ipotizzato
    P = stat.hypergeom(nn, M, n)
    a = P.pmf(r)*prob_priori(Nmin, Nmax)
    A = np.insert(A, len(A), a)

#dato che il passo degli N è unitario basta sommare
norm = sum(A)
A /= norm

plt.figure(1)
plt.title('Distribuzione andamento lupi')
plt.xlabel('numeor di lupi N')
plt.ylabel('P(N|Mnr)')
plt.grid()
plt.plot(N, A)
plt.show()