"""
Simple code that implements the nested sampling
to compute the evidence D-dimensiona gaussian.
The posterior distributions of the parameters are also calculated.
It was always assumed that the a priori distribution
was uniform according to the indifference principle.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

mu  = 0   # mean
sig = 0.9 #1.3 # standard deviation

def log_likelihood(x, D):
    """
    log likelihood of gaussian

    Parameters
    ----------
    x : 1darray
        array of parameters
    D : int
        dimension of parameter space

    Return
    ------
    likelihood : float
        log likelihood
    """
    
    likelihood = - np.log(2*np.pi*sig**2)*0.5*D - 0.5*np.sum((x - mu)**2)/sig**2  

    return likelihood


def samplig(x, D, bound, step, N_mcmc):
    """
    Sampling a new point of parameter space from
    a uniform distribution as proposal with the
    constraint to go up in likelihood

    Parameters
    ----------
    x : 1darray
        values of parameters and the relative likelihood
        x[:len(x)-1] = parameters
        x[len(x)-1] = likelihood(parameters)
    D : int
        dimension for parameter space, len(x) = D+1
    bound: float
        bounds of parameter space
    step : 1darray
        initial step for each dimension
    N_mcmc : int
        number of montecarlo steps

    Return
    ------
    new_sample : 1darray
        new array replacing x with higher likelihood
    accept : int
        number of moves that have been accepted
    reject : int
        number of moves that have been rejected
    """

    logLmin = x[D] # Worst likelihood
    point = x[:D]  # Point in the parameter space
    accept = 0     # Number of accepted moves
    reject = 0     # Number of rejected moves

    while True:
        # Array initialization
        new_sample = np.zeros(D+1)
        
        # Loop over the components
        for i in range(D):
            #we sample a trial variable with a gaussina distribution
            new_sample[i] = np.random.normal(point[i], step[i])
            
            # If it is out of bound...
            while new_sample[i] <= bound[2*i] or new_sample[i] >= bound[2*i + 1]:
                #...we resample the variable
                new_sample[i] = np.random.normal(point[i], step[i])

        # Computation of the likelihood associated to the new point
        new_sample[D] = log_likelihood(new_sample[:D], D)
        
        reject += 1
        # If the likelihood is greater we accept
        if new_sample[D] > logLmin :
            accept += 1
            reject -= 1 # To avoind one if check
            point[:D] = new_sample[:D]
            

            if accept > N_mcmc: #ACHTUNG
                """
                The samples must be independent. We trust
                that they are after N_mcmc accepted attempts,
                but we should compute the correlation of the D
                chains and the autocorrelation time in order
                to know what to write instead of N_mcmc, which is
                computationally expensive
                """
                break

        # We change the step to go towards a 50% acceptance
        if accept != 0 and reject != 0:
            if accept > reject :
                step *= np.exp(1.0 / accept);
            if accept < reject :
                step /= np.exp(1.0 / reject);

    return new_sample, accept, reject


def nested_samplig(N, D, bound, N_mcmc, tau=1e-6, verbose=False):
    """
    Compute evidence, information and distribution of parameters

    Parameters
    ----------
    N : int
        number of points
    D : int
         dimension for parameter space
    bound: float
        bounds of parameter space
    N_mcmc : int
        number of montecarlo steps
    tau : float
        tollerance, the run stops when the
        variation of evidence is smaller than tau
    verbose : bool, optional
        if True some information are printed during
        the execution to see what is happening

    Retunr
    ------
    calc : dict
        a dictionary which contains several information:
        "evidence"         : logZ,
        "error_lZ"         : error,
        "posterior"        : posterior,
        "worst_likelihood" : np.array(logL_list),
        "prior"            : prior_sample,
        "prior_mass"       : np.array(prior_mass),
        "number_acc"       : accepted,
        "number_rej"       : rejected,
        "number_steps"     : count,
        "log_information"  : np.array(logH_list),
        "list_evidence"    : np.array(logZ_list)

    """

    grid = np.zeros((N, D + 1)) # grid of live points

    prior_mass = [] # Integration variable
    logH_list  = [] # To store the information
    logZ_list  = [] # To store the evidence
    logL_list  = [] # To store the wrost likelihood
    all_value  = [] # To store all sample for the posterior for our parameter

    logH = -np.inf  # ln(Information, initially 0)
    logZ = -np.inf  # ln(Evidence Z, initially 0)

    # Indifference principle, the parameters' priors are uniform
    prior_sample = np.zeros((N, D))
    for i in range(D):
        prior_sample[:, i] = np.random.uniform(bound[2*i], bound[2*i + 1], size=N)

    #initialization of the parameters' values
    grid[:, :D] = prior_sample
    #likelihood initialization
    for i in range(N):
        grid[i, D] = log_likelihood(prior_sample[i,:], D)
    
    # Outermost interval of prior mass
    logwidth = np.log(1.0 - np.exp(-1.0/N))

    count    = 0   # Number of steps
    rejected = 0   # Total rejected steps
    accepted = 0   # Total accepted steps

    while True:
        count += 1                       # We refresh the number of steps
        prior_mass.append(logwidth)      # We store the integration variable

        Lw_idx = np.argmin(grid[:, D])   # Index for the parameters with the worst likelihood, i.e. the smallest one
        logLw = grid[Lw_idx, D]          # Value of the worst likelihood
        
        # Save wrost value for parameters' posterior; np.copy is crucial
        logL_list.append(logLw)
        all_value.append(np.copy(grid[Lw_idx, :D]))

        #np.logaddexp(x, y) = np.log(np.exp(x) + np.exp(y))
        logZnew = np.logaddexp(logZ, logwidth+logLw)

        logZ = logZnew           # We refresh the evidence
        logZ_list.append(logZ)   # We keep the value of the evidence

        # We compute the information and keep it
        logH = np.logaddexp(logH, logwidth + logLw - logZ + np.log(logLw - logZ))
        logH_list.append(logH)

        # New sample used to replace the points we have to delete
        # A good guess for initial step size is the tipical dimension of actual sample
        sampling_step = np.array([np.std(grid[:, jj]) for jj in range(D)])
        new_sample, acc, rej = samplig(grid[Lw_idx], D, bound, sampling_step, N_mcmc)
        accepted += acc # We refresh the total accepted steps
        rejected += rej # We refresh the total rejected steps

        grid[Lw_idx] = new_sample # Replacement
        logwidth -= 1.0/N         # Interval
        
        if verbose :
            # Evidence error computed each time for the print
            error = np.sqrt(np.exp(logH)/N)
            print(f"Iter = {count} acceptance = {accepted/(accepted+rejected):.3f} logZ = {logZ:.3f} error_logZ = {error:.3f} H = {np.exp(logH):.3f} \r", end="")

        if count > 3:
            #break
            if abs((logZ_list[-1] - logZ_list[-2])/logZ_list[-2]) < tau :
                break

    # Evidence error
    error = np.sqrt(np.exp(logH)/N)
    print(f"Iter = {count} acceptance = {accepted/(accepted+rejected):.3f} logZ = {logZ:.3f} error_logZ = {error:.3f} H = {np.exp(logH):.3f} ")
    
    # Compute posterior distributions
    N_resample = int(1e5)
    posterior  = np.zeros((N_resample, D))
    # Compute the weights
    W_sample  = np.exp(np.array(logL_list) + np.array(prior_mass) - logZ_list[-1])
    W_sample /= np.sum(W_sample)
    
    resample_idx = np.random.choice(range(count), N_resample, p=W_sample)
    for i in range(N_resample):
        posterior[i,:] = all_value[resample_idx[i]]
    
    calc = {
            "evidence"         : logZ,
            "error_lZ"         : error,
            "posterior"        : posterior,
            "worst_likelihood" : np.array(logL_list),
            "prior"            : prior_sample,
            "prior_mass"       : np.array(prior_mass),
            "number_acc"       : accepted,
            "number_rej"       : rejected,
            "number_steps"     : count,
            "log_information"  : np.array(logH_list),
            "list_evidence"    : np.array(logZ_list)
    }

    return calc


def plot_hist_par(prior, posterior, D, bound, save=False, show=False):
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
    bound : 1darray
        bound for plot
    save : bool, optional
        if True all plots are saved in the current directory
    show : bool, optional
        if True all plots are showed on the screen
    """
    
    for pr, ps, k, b_min, b_max in zip(prior.T, posterior.T, range(D), bound[0::2], bound[1::2]):
        
        fig = plt.figure(k+1)
        plt.title(f"Posterior for {k+1}-th parameter", fontsize=15)
        plt.xlabel("bound", fontsize=15)
        plt.ylabel("Probability density", fontsize=15)
        # normal distribution
        x = np.linspace(b_min, b_max, 1000)
        plt.plot(x, np.exp(-(x - mu)**2/(2*sig**2))/np.sqrt(2*np.pi*sig**2), 'r', label='normal')
        # prior and posterior
        plt.hist(pr, bins=50, density=True, histtype='step', color='blue', label='prior')
        plt.hist(ps, bins=50, density=True, histtype='step', color='black', label="posterior")
        plt.legend(loc='best')
        plt.grid()

        if save :
            plt.savefig(f"parametro{k+1}")
            plt.close(fig)

    if show :
        plt.show()


if __name__ == "__main__":

    np.random.seed(69420)
    
    # Number of points
    N = int(1e3)
    
    # Number of montecarlo steps
    N_mcmc = 20
    
    # Bound for our system: D-dimesional gaussian likelihood
    bound = np.array([-6, 6]*3)
    
    # Dimension of our parameter space
    D = len(bound)//2

    start = time.time()

    NS = nested_samplig(N, D, bound, N_mcmc, verbose=True)

    evidence        = NS["evidence"]
    error_evidence  = NS["error_lZ"]
    posterior_param = NS["posterior"]
    Wrost_L         = NS["worst_likelihood"]
    prior_param     = NS["prior"]
    prior_mass      = NS["prior_mass"]
    acc             = NS["number_acc"]
    rej             = NS["number_rej"]
    number_iter     = NS["number_steps"]
    log_information = NS["log_information"]
    list_evidence   = NS["list_evidence"]

    print(f"Evidence sampling    = {evidence:.3f} +- {error_evidence:.3f}")

    print(f"Number of iterations = {number_iter}")

    acceptance = acc/(acc+rej)
    print(f"Acceptance = {acceptance:.3f}")

    end = time.time() - start

    print(f"Elapsed time = {end//60:.0f} min and {end%60:.0f} s")

    plot_hist_par(prior_param, posterior_param, D, bound, show=True)
    
    plt.figure(1)
    plt.xlabel("Number of iteration", fontsize=15)
    plt.title("Trend of eìevidence and information")
    plt.plot(log_information, 'b', label='Information')
    plt.plot(list_evidence, 'k', label='Evidence')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
