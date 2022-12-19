#=
Simple code that implements the nested sampling
to compute the evidence of a multidimensional gaussian
=#
using Printf
using Random
using DelimitedFiles

Random.seed!(69420)


function log_likelihood(x, D, vec)
    #=
    log likelihood of gaussian

    Parameters
    ----------
    x : array or matrix
        array of parameters
    D : int
        dimension of parameter space
    vec : bool
        if True x must be a matrix NxD, this is convenient to
        initialize the likelihood at first step of nested_sampling
        if False x must be a D-dimensional array

    Return
    ------
    likelihood : array or float
        log likelihood of gaussian
        list if vec = True
        float if vec = False
    =#
    if vec == true
        N = length(x[:, 1])
        log_likelihood = Float64[]
        for i in 1:N
            v = x[i, :]
            g = - 0.5*D*log(2*pi) - 0.5*sum(v.^2)
            push!(log_likelihood, g)
        end
    else
        log_likelihood = - 0.5*D*log(2*pi) - 0.5*sum(x.^2)
    end
    
    return log_likelihood
    
end


function sampling(x, D, bound, thr)
    #=
    Sampling a new point of parameter space from
    a uniform distribution as proposal with the
    constraint to go up in likelihood

    Parameters
    ----------
    x : 1darray
        values of parameters and the relative likelihood
        x[1:D] = parameters
        x[D-1] = likelihood(parameters)
    D : int
        dimension for parameter space, len(x) = D+1
    bound: float
        bounds of parameter space
    thr : int
        threshold for acceptance for indipendent sampling

    Return
    ------
    new_sample : 1darray
        new array replacing x with higher likelihood
    accept : int
        number of moves that have been accepted
    reject : int
        number of moves that have been rejected
    =#
    local new_sample, accept, reject

    logLmin = x[D+1] #worst likelihood
    point = x[1:D]   #point in the parameter space
    step = 0.1       #initial step of the algorithm, to set
    accept = 0       #number of accepted moves
    reject = 0       #number of rejected moves
    
    while true
        #array initialization
        new_sample = Float64[]
        for i in 1:D+1
            push!(new_sample, 0.0)
        end
        #loop over the components
        for i in 1:D
            #we sample a trial variable
            draw =  step * (2*rand(Float64) - 1)
            new_sample[i] = point[i] + draw
            #...we resample the variable
            while abs(new_sample[i]) > bound
                #...we resample the variable
                draw =  step * (2*rand(Float64) - 1)
                new_sample[i] = point[i] + draw
            end
        end
        #computation of the likelihood associated to the new point
        new_sample[D+1] = log_likelihood(new_sample[1:D], D, false)

        #if the likelihood is smaller than before we reject
        if new_sample[D+1] < logLmin
            reject += 1
        
        #if greater we accept
        elseif new_sample[D+1] > logLmin
            accept += 1
            point[1:D] = new_sample[1:D]

            if accept > thr #ACHTUNG
                #=
                the samples must be independent. We trust
                that they are after thr accepted attempts,
                but we should compute the correlation of the D
                chains and the autocorrelation time in order
                to know what to write instead of thr, which is
                computationally expensive
                =#
                break
            end
        end
        
        # We change the step to go towards a 50% acceptance
        if accept != 0 && reject != 0
            if accept > reject 
                step *= exp(1.0 / accept)
            elseif accept < reject 
                step /= exp(1.0 / reject)
            end
        end

    end
    
    return new_sample, accept, reject

end


function log_sum(x, y)
    #=
    compute log(exp(x) + exp(y))
    numerically stable
    
    Parameter
    ---------
    x : float
        argument of first exp
    y : float
        argument of second exp
    
    Return
    ------
    sum : float
        log(exp(x) + exp(y))
    =#
    if x > y
        sum = x + log(1 + exp(y - x)) 
    else
        sum = y + log(1 + exp(x - y))
    end
    
    return sum
end


function nested_samplig(N, D, bound, tau, thr, verbose=false)
    #=
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
    thr : int
        threshold for acceptance for indipendent sampling
    verbose : bool, optional
        if True some information are printed during
        the execution to see what is happening

    Retunr
    ------
    calc : dict
        a dictionary which contains several information:
        "evidence"        => logZ,
        "error_lZ"        => error,
        "posterior"       => grid[:, :D],
        "likelihood"      => grid[:,  D],
        "prior"           => prior_sample,
        "prior_mass"      => np.array(prior_mass),
        "number_acc"      => accepted,
        "number_rej"      => rejected,
        "number_steps"    => Iter,
        "log_information" => np.array(logH_list),
        "list_evidence"   => np.array(logZ_list)
    =#
    local logZ, error
    grid = rand(Float64, (N, D+1))
    
    prior_mass = Float64[] #integration variable
    logH_list  = Float64[] #we keep the information
    logZ_list  = Float64[] #we keep the evidence

    logH = -Inf  # ln(Information, initially 0)
    logZ = -Inf  # ln(Evidence Z, initially 0)

    #indifference principle, the parameters' priors are uniform
    prior_sample = bound .* (2 .* rand(Float64, (N, D)) .- 1)

    #initialization of the parameters' values
    grid[:, 1:D] = prior_sample
    #likelihood initialization
    grid[:, D+1] = log_likelihood(prior_sample, D, true)
    
    # Outermost interval of prior mass
    logwidth = log(1.0 - exp(-1.0/N))

    Iter     = 0   #number of steps
    rejected = 0   #total rejected steps
    accepted = 0   #total accepted steps
    
    while true
        
        Iter += 1                        #we refresh the number of steps
        push!(prior_mass, logwidth)      #we keep the integration variable

        Lw_idx = argmin(grid[:, D+1])    #Index for the parameters with the
                                         #worst likelihood, i.e. the smallest one
        logLw = grid[Lw_idx, D+1]        #Value of the worst likelihood
        
        logZnew = log_sum(logZ, logwidth+logLw)

        logZ = logZnew           #we refresh the evidence
        push!(logZ_list, logZ)   #we keep the value of the evidence

        #we compute the information and keep it
        logH = log_sum(logH, logwidth + logLw - logZ + log(logLw - logZ))
        push!(logH_list, logH)

        #new sample used to replace the points we have to delete
        new_sample, acc, rej = sampling(grid[Lw_idx, :], D, bound, thr)
        accepted += acc #we refresh the total accepted steps
        rejected += rej #we refresh the total rejected steps
        
        grid[Lw_idx, :] = new_sample #replacement
        logwidth -= 1.0/N            #interval shrinking
        
        if verbose
            error = sqrt(exp(logH)/N)
            Acc = accepted/(accepted+rejected)
            @printf("Iter = %d, logZ = %0.5f +- %0.5f, acc = %0.3f \n", Iter, logZ, error, Acc)
        end
        
        if Iter > 3
            if abs(logZ_list[end] - logZ_list[end-1]) < tau
                break
            end
        end
    end
    #evidence error
    error = sqrt(exp(logH)/N)
    
    calc = Dict(
                "evidence"        => logZ,
                "error_lZ"        => error,
                "posterior"       => grid[:, 1:D],
                "likelihood"      => grid[:, D+1],
                "prior"           => prior_sample,
                "prior_mass"      => prior_mass,
                "number_acc"      => accepted,
                "number_rej"      => rejected,
                "number_steps"    => Iter,
                "log_information" => logH_list,
                "list_evidence"   => logZ_list
               )
    
    return calc
    
end

#number of point
N = Int(1e4)
#dimesion of space
D = 50
#integration limit, in these units it is the number of standard deviations
bound = 6
#tollerance of algorithm
tau = 1e-6
#threshold for acceptance must be tuned (17 D=50), (40 D=3)
thr = 17

@time NS = nested_samplig(N, D, bound, tau, thr, true)

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

#analitc result
logZ_v = -D*log(2*bound)

@printf("Evidence sampling    = %0.5f +- %0.5f \n", evidence, error_evidence)
@printf("Theoretical evidence = %0.5f \n", logZ_v)

@printf("Number of iterations = %d \n", number_iter)

acceptance = acc/(acc+rej)
@printf("Acceptance = %0.3f \n", acceptance)

#save some information
writedlm("param.txt", [N, D])
writedlm("prior.txt", prior_param)
writedlm("poste.txt", posterior_param)

    

