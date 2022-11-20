import time
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def log_posterior(x, **kwargs):
    """
    Distribution to sampling
    
    Parameters
    ----------
    x : float
        point of chain
    
    Returns
    -------
    logP : float
        log(posterior)
    
    Other Parameters
    ----------------
    **kwargs : dict of tuple, optional
        kwargs is a dictionary whose values
        ​​are tuples containing the optional
        arguments to pass to prior and likelihood
    """
    
    
    prior_args = kwargs['prior_args']
    likelihood_args = kwargs['likelihood_args']

    logP = log_prior(x, *prior_args) + log_likelihood(x, *likelihood_args)
    
    return logP


def log_prior(x, min=-10, max=10):
    """
    Prior distribution, we asumme uniform
    distribution between min and max
    
    Parameters
    ----------
    min, max : float, otional, dafult -10, 10
                |1   for min < x < max
        prior = |
                |0   otherwise
    
    Returns
    -------
    log(prior)
    """
    if x < min or x > max:
        #log(0)
        return -np.inf
    #log(1)
    return 0.0


def log_likelihood(x, mu=0.0, sigma=1.0):
    """
    likelihood
    
    Parameters
    ----------
    mu, sigma : float, otipnal, dafult 0, 1
        Gaussian parameters
    
    Returns
    -------
    scarto2 : float
        squared deviation from the
        model weighted with errors
    """
    
    scarto2 = ((x - mu)/sigma)**2
    
    return -0.5*scarto2


def uniform_proposal(x0, rng, min=-1, max=1):
    """
    Distribution that we know how to sample
    
    Parameters
    ----------
    x0 : float
        piont of the chain at 'time' i
    rng : Generator(PCG64)
        random number generator fomr numpy
    min, max : float, optional, dafult -1, 1
        ends of the support
        
    Returns
    ----------
    uniform distribution centered in x0
    """
    #da scegliere accuratamente i parametri dell'uniforme
    return x0 + rng.uniform(min, max)


def gaussian_proposal(x0, rng, mu=0, sigma=1):
    """
    Distribution that we know how to sample
    
    Parameters
    ----------
    x0 : float
        piont of the chain at 'time' i
    rng : Generator(PCG64)
        random number generator fomr numpy
    mu, sigma : float, otipnal, dafult 0, 1
        Gaussian parameters
        
    Returns
    ----------
    gaussian distribution centered in x0
    """
    #da scegliere accuratamente i parametri della gaussiana
    return x0 + rng.normal(mu, sigma)
    

def metropolis_hastings(target, proposal, rng, n=int(1e3), **kwargs):
    """
    Metropolis hastings algorithm
    
    Parameters
    ----------
    target : callable
        distribution to sampling
    proposal : callable
        Distribution that we know how to sample
    rng : Generator(PCG64)
        random number generator fomr numpy
    n : int
        number of iteration
    
    Returns
    -------
    samples : 1darray
        array of montecarlo chain
    
    Other Parameters
    ----------------
    **kwargs : dict of tuple and dict, optional
        kwargs is a dictionary whose values
        ​​are tuples containing the optional
        arguments to pass to target and proposal
        for target must use another dictonary
    """

    target_args = kwargs['target_args']     #dict
    proposal_args = kwargs['proposal_args'] #tupla
    
    samples = np.zeros(n)
    
    x0 = rng.uniform(-10, 10)
    logP0 = target(x0, **target_args)
    
    accepted = 0
    rejected = 1
    
    for i in range(n):
        
        x_pr = proposal(x0, rng, *proposal_args)
        logP = target(x_pr, **target_args)
        logr = logP - logP0 #rapporto delle probabilità
        
        if np.log(rng.uniform(0, 1)) < logr:
            x0 = x_pr
            logP0 = logP
            samples[i] = x_pr
            accepted += 1
        else:
            samples[i] = x0
            rejected += 1
        
        print(f"Iterazione numero: {i}, accetanza: {accepted/(accepted+rejected)} \r", end='')
    
    print(f"iterazione numero: {i}, accetanza: {accepted/(accepted+rejected)}")
    
    return samples


def autocorrelation(chain):
    """
    compute autocorrelation of the chain
    
    Paramateres
    -----------
    chain : 1darray
        array of montecarlo chain
    
    Returns
    -------
    auto_corr : 1darray
        array with auto-correlation of chain
    """

    m = np.mean(chain) #mean
    s = np.var(chain)  #variance
    
    xhat = chain - m
    auto_corr = np.correlate(xhat, xhat, 'full')[len(xhat)-1:]
    
    auto_corr = auto_corr/s/len(xhat) #normalizzation
    
    return auto_corr


def ACT(acf, tol=0.01, n=1):
    """
    Compute auto-correlation time finding
    the first zero of auto-correlation function
    
    Parameters
    ----------
    acf : 1darray
        array with auto-correlation of chain
    tol : float
        tollerance for the search of zero
    n : float
        expansion paramater that
        multiplies the first zero
    
    Returns
    -------
    time : float
        (first zero of auto-correlation function)*n
    """
    
    time = np.where(np.abs(acf) < tol)[0][0]#prendo il primo zero
    time *= n
    
    return time


if __name__ == "__main__":
    
    start = time.time()
    #seed
    rng = np.random.default_rng(69420)
    
    misure = int(5e4)    #misure di interesse
    termal = int(1e2)    #misure da scartare per la termalizzazione
    
    n = misure + termal  #numeero di misure totale da eseguire
    
    prior_args = ()                 #parametri della prior
    likelihood_args = (3, 0.5)      #parametri della likelihood
                                    #se vuota -> argomenti di defaut
    
    #passo gli argomenti  come un dizionario di dizionari per
    #poi spacchettarli più facilmente fra le varie funzioni
    target_args = {
                  'prior_args':prior_args,
                  'likelihood_args':likelihood_args
                  }
    
    #campionamento
    samples = metropolis_hastings(
              log_posterior, gaussian_proposal, rng, n=n,
              target_args=target_args,
              proposal_args=(0, 1)     #se vuota, argomenti di defaut
              )
    #calcolo auto-correlazione
    acf = autocorrelation(samples)
    
    
    plt.figure(1, figsize=(14, 6))
    plt.suptitle('Output data')
    
    plt.subplot(121)
    plt.title('chain')
    plt.plot(samples, '.b')
    plt.xlabel('iteration')
    plt.ylabel('samples')
    plt.grid()
    
    plt.subplot(122)
    plt.title('ACF')
    plt.plot(acf)
    plt.xlabel('iteration')
    plt.ylabel('auto-correlation function')
    plt.grid()
    
    #termalizzation, scarto le prime termal misure
    samples = samples[termal:]
    ac_time = ACT(acf, tol=0.01, n=1)
    print(f"Auto-correlation time = {ac_time}")
    
    #correlazione, prendo un elemento ogni ac_time
    samples = samples[::ac_time]
    
    plt.figure(2)
    plt.title('cleaned chain')
    plt.plot(samples, '.b')
    plt.xlabel('iteration')
    plt.ylabel('samples')
    plt.grid()
    
    #distribution
    pdf = norm(*likelihood_args)
                               #media +- 6*sigma per un grafico leggibile 
    x = np.linspace(likelihood_args[0] - 6*likelihood_args[1], 
                    likelihood_args[0] + 6*likelihood_args[1], 1000)
    
    plt.figure(3)
    plt.title('Distribution')
    plt.hist(samples, density=True, bins=int(np.sqrt(len(samples)-1)))
    plt.plot(x, pdf.pdf(x),'-k')
    plt.xlabel("x")
    plt.ylabel("pdf(x)")
    plt.grid()
    
    end = time.time() - start
    print(f'Elapsed time = {end:.3f} seconds')
    
    plt.show()