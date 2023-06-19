"""
Simple code that implements the nested sampling
to compute the evidence of a multidimensional gaussian
"""
import time
import numpy as np
import matplotlib.pyplot as plt


def log_likelihood(x, D):
    """
    log likelihood of gaussian

    Parameters
    ----------
    x : array or matrix
        array of parameters
    D : int
        dimension of parameter space
    vec : bool, optional
        if True x must be a matrix NxD, this is convenient to
        initialize the likelihood at first step of nested_sampling
        if False x must be a D-dimensional array

    Return
    ------
    likelihood : list or float
        log likelihood of gaussian
        list if vec = True
        float if vec = False
    """
    mu = 0
    likelihood =  - 0.5*D*np.log(2*np.pi) - 0.5*np.dot((x-mu), (x-mu))

    return likelihood


def prior(x, bound):
    if x < -bound or x > bound:
        return 0
    else :
        return 1


def samplig(x, D, bound, step):
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

    Return
    ------
    new_sample : 1darray
        new array replacing x with higher likelihood
    accept : int
        number of moves that have been accepted
    reject : int
        number of moves that have been rejected
    """

    logLmin = x[D] #worst likelihood
    point = x[:D]  #point in the parameter space
    #step = 0.1     #initial step of the algorithm, to set
    accept = 0     #number of accepted moves
    reject = 0     #number of rejected moves

    while True:
        #array initialization
        new_sample = np.zeros(D+1)
        #loop over the components
        #"""
        for i in range(D):
            #we sample a trial variable
            #new_sample[i] = np.random.normal(point[i], step[i])
            try_sample = point[i] + np.random.uniform(-step[i], step[i])
            r = prior(try_sample, bound)/prior(new_sample[i], bound)
            if np.random.uniform() < r:
                new_sample[i] = try_sample

            #if it is out of bound...
            #while np.abs(new_sample[i]) > bound:
                #...we resample the variable
            #    new_sample[i] = np.random.normal(point[i], step[i])
                #new_sample[i] = point[i] + np.random.uniform(-step[i], step[i])
        #"""
        #new_sample[:D] = np.random.uniform(-bound, bound, size=D)
        #computation of the likelihood associated to the new point
        new_sample[D] = log_likelihood(new_sample[:D], D)

        #if the likelihood is smaller than before we reject
        if new_sample[D] < logLmin:
            reject += 1
        #if greater we accept
        if new_sample[D] > logLmin:
            accept += 1
            point[:D] = new_sample[:D]

            if accept > 40:#ACHTUNG
                """
                the samples must be independent. We trust
                that they are after 40 accepted attempts,
                but we should compute the correlation of the D
                chains and the autocorrelation time in order
                to know what to write instead of 40, which is
                computationally expensive
                """
                break

        # We change the step to go towards a 50% acceptance
        #if accept != 0 and reject != 0:
        if accept > reject :
            step *= np.exp(1.0 / accept)
        if accept < reject :
            step /= np.exp(1.0 / reject)

    return new_sample, accept, reject


def nested_samplig(N, D, bound, tau=1e-6, verbose=False):
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
        "evidence"        : logZ,
        "error_lZ"        : error,
        "posterior"       : grid[:, :D],
        "likelihood"      : grid[:,  D],
        "prior"           : prior_sample,
        "prior_mass"      : prior_mass,
        "number_acc"      : accepted,
        "number_rej"      : rejected,
        "number_steps"    : iter,
        "log_information" : logH_list,
        "list_evidence"   : logZ_list

    """

    grid = np.zeros((N, D + 1))

    prior_mass = [] #integration variable
    logH_list  = [] #we keep the information
    logZ_list  = [] #we keep the evidence
    Sample_pts = []
    logLsa_pts = []

    logH = -np.inf  # ln(Information, initially 0)
    logZ = -np.inf  # ln(Evidence Z, initially 0)

    #indifference principle, the parameters' priors are uniform
    prior_sample = np.random.uniform(-bound, bound, size=(N, D))

    #initialization of the parameters' values
    grid[:, :D] = prior_sample
    #likelihood initialization
    for i in range(N):
        grid[i,  D] = log_likelihood(prior_sample[i,:], D)
    # Outermost interval of prior mass
    logwidth = np.log(1.0 - np.exp(-1.0/N))

    Iter     = 0   #number of steps
    rejected = 0   #total rejected steps
    accepted = 0   #total accepted steps

    while True:
        Iter += 1                        #we refresh the number of steps
        prior_mass.append(logwidth)      #we keep the integration variable

        Lw_idx = np.argmin(grid[:, D])   #index for the parameters with the worst likelihood, i.e. the smallest one
        logLw = grid[Lw_idx, D]          #value of the worst likelihood

        #np.logaddexp(x, y) = np.log(np.exp(x) + np.exp(y))
        logZnew = np.logaddexp(logZ, logwidth+logLw)

        logZ = logZnew           #we refresh the evidence
        logZ_list.append(logZ)   #we keep the value of the evidence

        #we compute the information and keep it
        logH = np.logaddexp(logH, logwidth + logLw - logZ + np.log(logLw - logZ))
        logH_list.append(logH)

        #Sample_pts.append(grid[Lw_idx, 0:-1])
        #logLsa_pts.append(grid[Lw_idx, 1])

        if N>1: # don't kill if n is only 1
            while True:
                copy = int(N * np.random.uniform()) % N  # force 0 <= copy < n
                if copy != Lw_idx:
                    break

        #logLstar = Obj[worst].logL;       # new likelihood constraint sarebbe logLw
        grid[Lw_idx] = grid[copy];           # overwrite worst object

        #new sample used to replace the points we have to delete
        sampling_step = np.array([np.std(grid[:, jj]) for jj in range(D)])
        #new_sample, acc, rej = samplig(grid[Lw_idx], D, bound, sampling_step)
        #while True:
        for tt in range(100):
            acc = 0
            rej = 0
            new_sample = np.zeros(D)
            new_sample = np.copy(grid[Lw_idx, 0:D])
            for ex in range(300):
                for i in range(D):
                #we sample a trial variable
                    #try_sample = np.random.normal(grid[Lw_idx, i], sampling_step[i])
                    try_sample = np.random.normal(new_sample[i], sampling_step[i])

                    r = prior(try_sample, bound)/prior(new_sample[i], bound)
                    if np.random.uniform() < r:
                        new_sample[i] = try_sample
                        acc += 1
                    else :
                        rej += 1
                #Sample_pts.append(new_sample)
                """
                if acc > rej :
                    sampling_step[i] *= np.exp(1.0 / acc)
                if acc < rej :
                    sampling_step[i] /= np.exp(1.0 / rej)
                """

                logL_p = log_likelihood(new_sample, D)
                Sample_pts.append(new_sample)
                logLsa_pts.append(logL_p)
                if logL_p > logLw :
                    grid[Lw_idx, 0:D] = new_sample
                    grid[Lw_idx, D] = logL_p
                    accepted += 1
                    break
                else :
                    rejected += 1

            if accepted > rejected :
                sampling_step *= np.exp(1.0 / accepted)
            if accepted < rejected :
                sampling_step /= np.exp(1.0 / rejected)

        #accepted += acc #we refresh the total accepted steps
        #rejected += rej #we refresh the total rejected steps

        #grid[Lw_idx] = new_sample #replacement
        logwidth -= 1.0/N         #interval shrinking

        if verbose :
            #evidence error computed each time for the print
            error = np.sqrt(np.exp(logH)/N)
            print(f"Iter = {Iter} acceptance = {accepted/(accepted+rejected):.3f} logZ = {logZ:.3f} error_logZ = {error:.3f} H = {np.exp(logH):.3f} \r", end="")

        if max(grid[:,D]) + logwidth < np.log(0.001) + logZ:
            break

    #evidence error
    error = np.sqrt(np.exp(logH)/N)

    w = np.exp([ll - logZ for ll in logLsa_pts])
    w = w/sum(w)
    calc = {
            "evidence"        : logZ,
            "error_lZ"        : error,
            "posterior"       : np.array(Sample_pts),#grid[:, :D],
            "likelihood"      : grid[:,  D],
            "prior"           : prior_sample,
            "prior_mass"      : np.array(prior_mass),
            "number_acc"      : accepted,
            "number_rej"      : rejected,
            "number_steps"    : Iter,
            "log_information" : np.array(logH_list),
            "list_evidence"   : np.array(logZ_list),
            "pesi"            : w
    }

    return calc


def plot_hist_par(prior, posterior, w, D, save=False, show=False):
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
    save : bool, optional
        if True all plots are saved in the current directory
    show : bool, optional
        if True all plots are showed on the screen
    """
    x = np.linspace(-5, 5, 100)
    for pr, ps, k in zip(prior.T, posterior.T, range(D)):

        fig = plt.figure(k+1)
        plt.title(f"Confronto per il {k+1}esimo parametro")
        plt.xlabel("bound")
        plt.ylabel("Probability density")
        plt.hist(pr, bins=50, density=True, histtype='step', color='blue', label='prior')
        plt.hist(ps, bins=50, density=True, weights=w, histtype='step', color='black', label="posterior")
        plt.plot(x, 1/(2*np.pi)**(1/(2*D)) * np.exp(-x**2/2))
        plt.legend(loc='best')
        plt.grid()

        if save :
            plt.savefig(f"parametro{k+1}")
            plt.close(fig)

    if show :
        plt.show()


if __name__ == "__main__":

    np.random.seed(69420)
    #number of points
    N = int(4e3)
    #dimesion
    D = 1
    #integration limit, in these units it is the number of standard deviations
    bound = 5

    start = time.time()

    NS = nested_samplig(N, D, bound, verbose=True)

    evidence        = NS["evidence"]
    error_evidence  = NS["error_lZ"]
    posterior_param = NS["posterior"]
    likelihood      = NS["likelihood"]
    prior_param     = NS["prior"]
    prior_mass      = NS["prior_mass"]
    acc             = NS["number_acc"]
    rej             = NS["number_rej"]
    number_iter     = NS["number_steps"]
    log_information = NS["log_information"]
    list_evidence   = NS["list_evidence"]
    pesi            = NS["pesi"]

    print() #print for problem in refresh of verbose in ubuntu shell
    print(f"Evidence sampling    = {evidence:.3f} +- {error_evidence:.3f}")
    print(f"Theoretical evidence = {-D*np.log(2*bound):.3f}")

    print(f"Number of iterations = {number_iter}")

    acceptance = acc/(acc+rej)
    print(f"Acceptance = {acceptance:.3f}")

    end = time.time() - start

    print(f"Elapsed time = {end//60:.0f} min and {end%60:.0f} s")

    plot_hist_par(prior_param, posterior_param, pesi, D, show=True)