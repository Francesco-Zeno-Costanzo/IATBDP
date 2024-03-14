
#=
using StatsBase
using PythonPlot
using Statistics
using FileIO
using JLD2
=#

function polynomial(p::Vector, x::Vector)

        curve = zeros(length(x))

        for (j, xi) in enumerate(x)
            pol = [p[i]*xi^(i-1) for i in 1:length(p)]
            pol = sum(pol)
            curve[j] = pol
        end

        return curve
    end

function plot_hist_par(prior_u::Matrix, posterior_u::Matrix, prior_n::Matrix, posterior_n::Matrix; save_fig::Bool=false)

    for k in 1:length(prior_n[1, :])

        fig = figure(k)

        title("Posterior for $(k)-th parameter", fontsize=15)
        xlabel("bound", fontsize=15)
        ylabel("Probability density", fontsize=15)

        # prior and posterior
        hist(prior_u[:, k],     bins=50, density=true, histtype="step", color="blue",  label="prior uni")
        hist(posterior_u[:, k], bins=50, density=true, histtype="step", color="black", label="posterior uni")
        hist(prior_n[:, k],     bins=50, density=true, histtype="step", color="green",  label="prior norm")
        hist(posterior_n[:, k], bins=50, density=true, histtype="step", color="red", label="posterior norm")
        legend(loc="best")
        grid()

        if save_fig
            savefig("parameter$(k)")
            plotclose(fig)
        end
    end

end

# Read data of nested samplind
data_uni  = load("Result_uni.jld2")
data_norm = load("Result_norm.jld2")

# Read data
data   = readdlm("data.txt", '\t')
x_data = data[2:length(data[:, 1]), 1]
y_data = data[2:length(data[:, 1]), 2]
dy     = ones(length(y_data))

plot_hist_par(data_uni["prior"], data_uni["posterior"], data_norm["prior"], data_norm["posterior"], save_fig=true)

D = length(data_uni["prior"][1, :])
N = length(data_uni["prior"][:, 1])

fig=figure(D+1)
title("Posterior predictive check", fontsize=15)
xlabel("x data [a.u.]", fontsize=15)
ylabel("y data [a.u.]", fontsize=15)

p_u = []
p_n = []
for i in 1:D # compute mean value of each posterior

    m_p = mean(data_uni["posterior"][:, i])
    s_p = std(data_uni["posterior"][:, i])/sqrt(N)
    println("$i-th parameter = $(round(m_p, digits=5)) +- $(round(s_p, digits=5))")
    push!(p_u, m_p)

    m_p = mean(data_norm["posterior"][:, i])
    s_p = std(data_norm["posterior"][:, i])/sqrt(N)
    println("$i-th parameter = $(round(m_p, digits=5)) +- $(round(s_p, digits=5))")
    push!(p_n, m_p)

end

t = collect(minimum(x_data):0.01:maximum(x_data))
errorbar(x_data, y_data, dy, fmt='.', label="data")
plot(t, polynomial(p_u, t), 'b', label="NS fit unifrom prior")
plot(t, polynomial(p_n, t), 'r', label="NS fit gaussian prior")
legend(loc="best")
grid()
savefig("PPC")
plotclose(fig)
plotshow()
