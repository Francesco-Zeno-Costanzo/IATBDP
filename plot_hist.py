"""
code for plot of ditribution 
calculated with julia code
"""
import time
import numpy as np
import matplotlib.pyplot as plt


def plot_hist_par(prior, posterior, N,  D, save=False, show=False):
    """
    Plot of posterior vs prior for all parameters

    Parameters
    ----------
    prior : 2darray
        matrix which contains the prior of all parameters
    posterior : 2darray
        matrix which contains the posterior of all parameters
    D : int
        dimension of the parameter space
    N : int
        number of point
    save : bool, optional
        if True all plots are saved in the current directory
    show : bool, optional
        if True all plots are showed on the screen
    """
    for pr, ps, k in zip(prior.T, posterior.T, range(D)):

        fig = plt.figure(k+1)
        plt.title(f"Confronto per il {k+1}esimo parametro")
        plt.xlabel("bound")
        plt.ylabel("Probability density")
        plt.hist(pr, bins=int(np.sqrt(N-1)), density=True, histtype='step', color='blue', label='prior')
        plt.hist(ps, bins=int(np.sqrt(N-1)), density=True, histtype='step', color='black', label="posterior")
        plt.legend(loc='best')
        plt.grid()

        if save :
            plt.savefig(f"plot_D_50/parametro{k+1}")
            plt.close(fig)

    if show :
        plt.show()


if __name__ == "__main__":

    start = time.time()

    #parameter of calculation
    param = np.loadtxt("param.txt", unpack=True, dtype=int)
    N, D = param
    
    #parameter distribution
    prior = np.zeros((N, D))
    poste = np.zeros((N, D))
    for i in range(N):
        #each row is the distribution of i-th parameter
        prior[i, :] = np.loadtxt("prior.txt", skiprows=i, max_rows=1)
        poste[i, :] = np.loadtxt("poste.txt", skiprows=i, max_rows=1)
    
    end = time.time() - start
    print(f"Time for read and save data  = {end:.3f} seconds ")
    
    start = time.time()
    
    plot_hist_par(prior, poste, N, D, save=True)
    
    end = time.time() - start
    print(f"Time for make and save plots = {end:.3f} seconds")
    
