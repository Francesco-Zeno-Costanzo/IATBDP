# IATBDP
Code repository for the Introduction to Bayesian Probability Theory course

Written together with Carlo Panu

# Brief explanation of the codes

## fit\_pol\_ran.py

Code that tries to fit the data contained in data.txt with a polynomial model looking for the optimal values ​​of the likelihood calculated in a set of parameters uniformly extracted in a range chosen by the user.

## lupi.py

Exercise: Having seen M wolves, having tagged them and recognizing r tagged wolves out of n observed at a later time, estimate the total population N.

## metropolis_hastings.py

Sampling a Gaussian using the metropolis-hastings algorithm

## SIRD.py

Simulation of SIRD model

## nested_sampling.py

Simple code that implements the nested sampling to compute the evidence D-dimensiona gaussian.
The posterior distributions of the parameters are also calculated.
It was always assumed that the a priori distribution was uniform according to the indifference principle.

## fit_nested_sampling.jl

Code for fitting data using nested sampling.
In this case, as an example, we wanted to use the same data for the fit\_pol\_ran.py code
and also the same function as a theoretical model.
Any changes for other models are not particularly complicated.
The calculations of the average values ​​of the parameters are calculated starting from the
various posterior distributions for the sole purpose of carrying out a posterior predictive
check.
It was always assumed that the a priori distribution was uniform,
according to the indifference principle.

