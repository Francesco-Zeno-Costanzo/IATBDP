#============================================================================================
Code for fitting data using nested sampling.
In this case, as an example, we wanted to use the same data for the fit_plo_ran.py code
and also the same function as a theoretical model.
Any changes for other models are not particularly complicated.
The calculations of the average values ​​of the parameters are calculated starting from the
various posterior distributions for the sole purpose of carrying out a posterior predictive
check
============================================================================================#

#using Random
#using StatsBase
#using PythonPlot
#using Statistics
#using DelimitedFiles

#============================================================================================#

int(x) = floor(Int, x) # int function, e.g. int(2.3) = 2

#============================================================================================#

@doc raw"""
Returns a polynomial of arbitrary
degree calculated on given points

Parameters
----------
p : 1darray,
    array containing the values ​​of the coefficients \
x : 1darray,
    data 

Returns
----------
curve : 1darray
    polynomial

Example
-------
julia> polinomio([0, 0, 1], [0, 1, 2, 3, 4]) # 1*x**2 + 0*x + 0
5-element Vector{Float64}:
  0.0
  1.0
  4.0
  9.0
 16.0
"""
function polynomial(p::Vector, x::Vector)
    
    n     = length(x)
    curve = zeros(n)
    num_par = length(p)
    
    for (j, xi) in enumerate(x)
        pol = [p[i]*xi^(i-1) for i in 1:num_par]
        pol = sum(pol)
        curve[j] = pol
    end
    
    return curve
end

#============================================================================================#
   
@doc raw"""
log likelihood of our data

Parameters
----------
x_data : 1darray,
    data point on x axis \
y_data : 1darray,
    data point on y axis \
dy : 1darray,
    error on y data \
x : 1darray, 
    array of parameters \
D : int, 
    dimension of parameter space \

Return
------
likelihood : float, 
    log likelihood
"""
function log_likelihood(x_data::Vector, y_data::Vector, dy::Vector, x::Vector, D::Int)
    
    likelihood = -0.5*sum( ((y_data .- polynomial(x, x_data)) ./ dy) .^ 2)

    return likelihood
end

#============================================================================================#

@doc raw"""
Sampling a new point of parameter space from
a uniform distribution as proposal with the
constraint to go up in likelihood

Parameters
----------
x_data : 1darray,
    data point on x axis \
y_data : 1darray,
    data point on y axis \
dy : 1darray,
    error on y data \
x : 1darray, 
    values of parameters and the relative likelihood \
    x[:len(x)-1] = parameters \
    x[len(x)-1] = likelihood(parameters) \
D : int, 
    dimension for parameter space, len(x) = D+1 \
bound: float, 
    bounds of parameter space \
step : 1darray, 
    initial step for each dimension \
N_mcmc : int, 
    number of montecarlo steps \

Return
------
new_sample : 1darray, 
    new array replacing x with higher likelihood \
accept : int, 
    number of moves that have been accepted \
reject : int, 
    number of moves that have been rejected \
"""
function sampling(x_data::Vector, y_data::Vector, dy::Vector, x::Vector, D::Int, bound::Vector, step::Vector, N_mcmc::Int)

    logLmin = x[D+1] # Worst likelihood
    point = x[1:D]   # Point in the parameter space
    accept = 0       # Number of accepted moves
    reject = 0       # Number of rejected moves
    
    # Array initialization
    new_sample = zeros(D+1)
        
    while true
        
        # Loop over the components
        for i in 1:D
            #we sample a trial variable with a gaussina distribution
            new_sample[i] = point[i] + randn()*step[i]
            
            # If it is out of bound...
            while new_sample[i] <= bound[2*i - 1] || new_sample[i] >= bound[2*i]
                #...we resample the variable
                new_sample[i] = point[i] + randn()*step[i]
            end
        end

        # Computation of the likelihood associated to the new point
        new_sample[D+1] = log_likelihood(x_data, y_data, dy, new_sample[1:D], D)
        
        reject += 1
        # If the likelihood is greater we accept
        if new_sample[D+1] > logLmin
            accept += 1
            reject -= 1 # To avoind one if check
            point[1:D] = new_sample[1:D] 

            if accept > N_mcmc # ACHTUNG:
                #===================================================
                The samples must be independent. We trust
                that they are after N_mcmc accepted attempts,
                but we should compute the correlation of the D
                chains and the autocorrelation time in order
                to know what to write instead of N_mcmc, which is
                computationally expensive
                ===================================================#
                break
            end
        end    

        # We change the step to go towards a 50% acceptance
        if accept != 0 && reject != 0
            if accept > reject
                step *= exp(1.0 / accept)
            else
                step /= exp(1.0 / reject)
            end
        end
    
    end
    
    return new_sample, accept, reject
end

#============================================================================================#

@doc raw"""
Function to compute: log(e^x + e^y) in smart way

Parameter
---------
x : float \
y : float \

Return
------
log(e^x + e^y)
"""
function logaddexp(x::Float64, y::Float64)
    
    if x > y 
        return x + log(1 + exp(y - x)) 
    else
        return y + log(1 + exp(x - y))
    end
end

#============================================================================================#

@doc raw"""
Compute evidence, information and distribution of parameters

Parameters
----------
x_data : 1darray,
    data point on x axis \
y_data : 1darray,
    data point on y axis \
dy : 1darray,
    error on y data \
N : int, 
    number of points \
D : int, 
    dimension for parameter space \
bound: float, 
    bounds of parameter space \
N_mcmc : int, 
    number of montecarlo steps \
tau : float, 
    tollerance, the run stops when the 
    variation of evidence is smaller than tau \
verbose : bool, optional, dafult false, 
    if True some information are printed during
    the execution to see what is happening

Return
------
calc : dict, 
    a dictionary which contains several information: \
    "evidence"         : logZ, \
    "error_lZ"         : error, \
    "posterior"        : posterior, \
    "worst_likelihood" : logL_list,  \
    "prior"            : prior_sample, \
    "prior_mass"       : prior_mass, \
    "number_acc"       : accepted, \
    "number_rej"       : rejected, \
    "number_steps"     : count, \
    "log_information"  : logH_list, \
    "list_evidence"    : logZ_list \

"""
function nested_sampling(x_data::Vector, y_data::Vector, dy::Vector, N::Int, D::Int, bound::Vector, N_mcmc::Int; tau::Float64=1e-6, verbose::Bool=false)

    grid = zeros(N, D + 1) # grid of live points

    prior_mass = [] # Integration variable
    logH_list  = [] # To store the information
    logZ_list  = [] # To store the evidence
    logL_list  = [] # To store the wrost likelihood
    all_value  = [] # To store all sample for the posterior for our parameter

    logH = -1e150 # ln(Information, initially 0)
    logZ = -1e150 # ln(Evidence Z, initially 0)

    # Indifference principle, the parameters' priors are uniform
    prior_sample = zeros(N, D)
    for i in 1:D
        prior_sample[:, i] = (bound[2*i] - bound[2*i - 1]) .* rand(N) .+ bound[2*i - 1]
    end
    
    #initialization of the parameters' values
    grid[:, 1:D] = prior_sample
    #likelihood initialization
    for i in 1:N
        grid[i, D+1] = log_likelihood(x_data, y_data, dy, prior_sample[i,:], D)
    end
    
    # Outermost interval of prior mass
    logwidth = log(1.0 - exp(-1.0/N))

    count    = 0   # Number of steps
    rejected = 0   # Total rejected steps
    accepted = 0   # Total accepted steps

    while true
        count += 1                    # We refresh the number of steps
        push!(prior_mass, logwidth)   # We store the integration variable

        Lw_idx = argmin(grid[:, D + 1])   # Index for the parameters with the worst likelihood, i.e. the smallest one
        logLw  = grid[Lw_idx, D + 1]      # Value of the worst likelihood
        
        # Save wrost value for parameters' posterior; copy is crucial
        push!(logL_list, logLw)
        push!(all_value, copy(grid[Lw_idx, 1:D]))

        logZnew = logaddexp(logZ, logwidth+logLw)

        logZ = logZnew           # We refresh the evidence
        push!(logZ_list, logZ)   # We keep the value of the evidence

        # We compute the information and keep it
        logH = logaddexp(logH, logwidth + logLw - logZ + log(logLw - logZ))
        push!(logH_list, logH)

        # New sample used to replace the points we have to delete
        # A good guess for initial step size is the tipical dimension of actual sample
        sampling_step = [std(grid[:, jj]) for jj in 1:D]
        new_sample, acc, rej = sampling(x_data, y_data, dy, grid[Lw_idx, :], D, bound, sampling_step, N_mcmc)
        accepted += acc # We refresh the total accepted steps
        rejected += rej # We refresh the total rejected steps

        grid[Lw_idx, :] = new_sample # Replacement
        logwidth -= 1.0/N            # Interval shrinking
        
        if verbose
            # Rounded at nc digits
            nc    = 3 # number of digits
            error = round(sqrt(exp(logH)/N), digits=nc)
            pracc = round(accepted/(accepted+rejected), digits=nc)
            plogZ = round(logZ, digits=nc)
            pinfo = round(exp(logH), digits=nc)
            print("Iter = $count acceptance = $pracc logZ = $plogZ error_logZ = $error H = $pinfo \r")
        end
        
        if count > 3
            #break
            if abs((logZ_list[count-1] - logZ_list[count-2])/logZ_list[count-2]) < tau
                break
            end
        end
    end
    
    # We want that the last line stay printed
    nc    = 3 # number of digits
    error = round(sqrt(exp(logH)/N), digits=nc)
    pracc = round(accepted/(accepted+rejected), digits=nc)
    plogZ = round(logZ, digits=nc)
    pinfo = round(exp(logH), digits=nc)
    println("Iter = $count acceptance = $pracc logZ = $plogZ error_logZ = $error H = $pinfo")
    
    # Evidence error
    error = sqrt(exp(logH)/N)
    
    # Compute posterior distributions
    N_resample = int(1e5)
    posterior  = zeros(N_resample, D)
    # Compute the weights
    W_sample  = exp.(logL_list .+ prior_mass .- logZ_list[count])
    W_sample /= sum(W_sample)
    
    for i in 1:N_resample
        resample_idx = sample(1:count, ProbabilityWeights(W_sample))
        posterior[i,:] = all_value[resample_idx]
    end
    
    calc = Dict(
            "evidence"         => logZ,
            "error_lZ"         => error,
            "posterior"        => posterior,
            "worst_likelihood" => logL_list,
            "prior"            => prior_sample,
            "prior_mass"       => prior_mass,
            "number_acc"       => accepted,
            "number_rej"       => rejected,
            "number_steps"     => count,
            "log_information"  => logH_list,
            "list_evidence"    => logZ_list
    )

    return calc
end

#============================================================================================#

@doc raw"""
Plot of posterior vs prior for all parameters

Parameters
----------
prior : 2darray, 
    matrix which contains the prior of all parameters \
posterior : 2darray, 
    matrix which contains the posterior of all parameters \
D : int, 
    dimension of the parameter space \
bound : 1darray , 
    bound for plot \
save : bool, optional, dafult false, 
    if True all plots are saved in the current directory without being shown on the screen
"""
function plot_hist_par(prior, posterior, D, bound; save=false)
    
    for k in 1:D
       
        fig = figure(k)
        
        title("Posterior for $(k)-th parameter", fontsize=15)
        xlabel("bound", fontsize=15)
        ylabel("Probability density", fontsize=15)
        
        # prior and posterior
        hist(prior[:, k],     bins=50, density=true, histtype="step", color="blue",  label="prior")
        hist(posterior[:, k], bins=50, density=true, histtype="step", color="black", label="posterior")
        legend(loc="best")
        grid()
        
        if save
            savefig("parametro$(k)")
            plotclose(fig)
        end
    end
     
end

#============================================================================================#

function main()

    Random.seed!(69420)
    
    # Number of points
    N = int(1e4)
    
    # Number of montecarlo steps
    N_mcmc = 20
    
    # Read data
    data   = readdlm("data.txt", '\t')
    x_data = data[2:length(b[:, 1]), 1]
    y_data = data[2:length(b[:, 1]), 2]
    dy     = ones(length(y_data))
    
    # Bound for our system:
    # Linear case:    log(Z) \simew -90
    #bound = [-9, -5, 1.5, 2.5]
    # Quadratic case: log(Z) \simeq -16
    bound = [-2, 2, -0.5, 0.5, 0, 0.2]
    # Cubic case:     log(Z) \simeq -19
    #bound = [-0.5, 1.5, -0.2, 0.2, 0, 0.1, -0.05, 0.05]
    
    # Dimension of our parameter space
    D = int(length(bound)/2)

    NS = nested_sampling(x_data, y_data, dy, N, D, bound, N_mcmc, verbose=true)

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

    println("Evidence sampling    = $evidence +- $error_evidence")

    println("Number of iterations = $number_iter")

    acceptance = acc/(acc+rej)
    println("Acceptance = $acceptance")

    plot_hist_par(prior_param, posterior_param, D, bound, save=false)
    
    #===================== PLOT =====================#
    
    figure(0)
    title("Trend of eìevidence and information", fontsize=15)
    xlabel("Number of iteration", fontsize=15)
    plot(log_information, color="blue", label="Information")
    plot(list_evidence, color="black", label="Evidence")
    legend(loc="best")
    grid()
    
    figure(D+1)
    title("Posterior predictive check", fontsize=15)
    xlabel("x data [a.u.]", fontsize=15)
    ylabel("y data [a.u.]", fontsize=15)
    
    p = []
    for i in 1:D # compute mean value of each posterior
        m_p = sum(posterior_param[:, i])/length(posterior_param[:, i])
        println("$i-th parameter = $m_p")
        push!(p, m_p)
    end
    
    t = minimum(x_data):0.01:maximum(x_data)
    errorbar(x_data, y_data, dy, fmt='.', label="data")
    plot(t, polynomial(p, t), 'b', label="NS fit")
    legend(loc="best")
    grid()
    
    plotshow()

end

@time main()
