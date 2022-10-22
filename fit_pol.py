import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def polinomio(p, x, num_par):
    """
    restituisce un polinomio di grado
    arbistrario calcolato per punti dati

    Parameters
    ----------
    p : 1darray
        array che contiene i valori dei coefficenti
    x : 1darray
        array su cui calcolare il polinomio
    num_par : int
        numero di parametri = grado del polinomio + 1

    Returns
    ----------
    curva : 1darray
        array che contine il polinomio

    esempio:
    >>> import numpy as np
    >>> x=np.linspace(0, 4, 5)
    >>> x
    array([0., 1., 2., 3., 4.])
    >>> polinomio([0, 0, 1], x, 3) # 1*x**2 + 0*x + 0
    array([ 0.,  1.,  4.,  9., 16.])
    """

    n = len(x)
    curva = np.zeros(n)

    for j, xi in enumerate(x):
        pol = [p[i]*xi**i for i in range(num_par)]
        pol = np.array(pol).sum()
        curva[j] = pol

    return curva


def log_likelihood(x, y, sigma, p, num_par):
    """
    restituisce lo scarto quadratico dei dati
    dal modelo pesato con gli errori

    Parameters
    ----------
    x : 1darray
        dati sulle x
    y : 1darray
        dati sulle y
    dy : 1darray
        errori dati sulle y
    num_par : int
        numero di parametri = grado del polinomio + 1

    Returns
    ----------
    scarto : float
        scarto quadratico dei dati dal modelo pesato con gli errori

    """

    modello = polinomio(p, x, num_par)
    scarto2 = -0.5*(((y - modello)/sigma)**2).sum()

    return scarto2


def test(x, y, dy, N, nmax, range_params, num_par):
    """
    Punzione che calcola i parametri ottimali di fit e l'evidenza del modello
    Parameters
    ----------
    x : 1darray
        dati sulle x
    y : 1darray
        dati sulle y
    dy : 1darray
        errori dati sulle y
    N : int
        numero di n-uple di parametri da generare
    nmax : int
        quante n-uple di parametri restituire
    range_params : ndarray
        matrice contentente i limiti dei parametri del tipo
        a_n^{max} = range_params[0, 1]; a_n^{min} = range_params[0, 0]
    num_par : int
        numero di parametri = grado del polinomio + 1

    Returns
    ----------
    popt : ndarray
        sottomatrice di range_params contenete nmax
        n-uple di parametri con likelihood maggiore

    """

    #genero set di parametri distribuiti uniformemente
    Params = np.zeros((N, num_par))
    for i in range(num_par):
        Params[:,i] = np.random.uniform(range_params[i,0], range_params[i, 1], N)

    #calcolo il lod della distibuzione a posteriori
    logP = np.zeros(N)
    for i in range(N):
        logP[i] = log_likelihood(x, y, dy, Params[i,:], num_par)


    #grafici in casi facili
    if num_par == 2:
        fig = plt.figure(0)
        ax = fig.add_subplot()
        S = plt.scatter(Params[:,1], Params[:,0], c=logP)
        plt.colorbar(S)
        ax.set_title('logaritmo likelihood')
        ax.set_xlabel('a1')
        ax.set_ylabel('a0')


    if num_par == 1:
        fig = plt.figure(0)
        plt.bar(Params[:,0], logP)
        plt.title('logaritmo likelihood')
        plt.xlabel('a0')

    #trovo le nmax n_uple di parametri che massimizzano la likelihood
    index = np.argsort(logP)[::-1]
    index = index[:nmax]

    popt = np.zeros((nmax, num_par))
    for i in range(num_par):
        popt[:,i] = Params[index, i]

    #calcolo dell'evidenza
    #np.log(np.sum(np.exp(logP)))
    Z = logsumexp(logP)

    for i in range(num_par):
        dx = (range_params[i, 1] - range_params[i, 0])/N
        Z += np.log(dx)

    return popt, Z



if __name__ == "__main__":

    x, y = np.loadtxt(r'C:\Users\franc\Documents\GitHub\IATBDP\data.txt', skiprows=1, unpack=True)
    dy = np.ones(len(y))

    print('fit con polinomio del tipo: a_n * x^n + ... + a_1 * x + a_0')
    gra_pol = int(input('scegliere grado del polinomio:'))
    num_par = gra_pol + 1

    print('Inserire il range in cui far variare i parametri')

    a_min = np.zeros(num_par)
    a_max = np.zeros(num_par)
    for i in range(num_par):
        a_min[i] = float(input(f'minimo  per il parametro a{abs(i-gra_pol)}:'))
        a_max[i] = float(input(f'massimo per il parametro a{abs(i-gra_pol)}:'))

    c = [a_min, a_max]
    c = np.array(c)
    c = c.T[::-1]

    n_curve = 3
    popt, evidence = test(x, y, dy, int(5e4), n_curve, c, num_par)
    print(f"log Evidence = {evidence}")

    plt.figure(1)
    plt.title('postirior predictive check')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()

    plt.errorbar(x, y, dy, fmt='.')

    for i in range(len(popt[:,0])):
        y_m = polinomio(popt[i, :], x, len(c[:,0]))
        plt.plot(x, y_m)
        print(popt[i,:])

    plt.show()