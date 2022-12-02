"""
Sir model, and variants, integrated with
Adams-Bashforth-Moulton
predictor and corretor of order 4
"""
import numpy as np
import matplotlib.pyplot as plt


def SIRD(t, Y, beta, gamma, delta):
    """
    equation to solve

    Parameters
    ----------
    t : float
        time
    Y : 1darray
        array of variables
    beta, gamma, delta : float
        model's parameters

    Return
    ------
    Y_dot : 1darray
        array of equations
    """
    #incognite
    S, I, R, D = Y
    #equazioni da risolvere
    S_dot = -beta*I*S/N
    I_dot =  beta*I*S/N - gamma*I - delta*I
    R_dot = gamma*I
    D_dot = delta*I

    Y_dot = np.array([S_dot, I_dot, R_dot, D_dot])

    return Y_dot


def AMB4(num_steps, tf, f, init, args=()):
    """
    Integrator with Adams-Bashforth-Moulton
    predictor and corretor of order 4

    Parameters
    ----------
    num_steps : int
        number of point of solution
    tf : float
        upper bound of integration
    f : callable
        function to integrate, must accept vectorial input
    init : 1darray
        array of initial condition
    args : tuple, optional
        extra arguments to pass to f

    Return
    ------
    X : array, shape (num_steps + 1, len(init))
        solution of equation
    t : 1darray
        time
    """
    #time steps
    dt = tf/num_steps

    X = np.zeros((num_steps + 1, len(init))) #matrice delle soluzioni
    t = np.zeros(num_steps + 1)              #array dei tempi

    X[0, :] = init                           #condizioni iniziali

    #primi passi con runge kutta
    for i in range(3):
        xk1 = f(t[i], X[i, :], *args)
        xk2 = f(t[i] + dt/2, X[i, :] + xk1*dt/2, *args)
        xk3 = f(t[i] + dt/2, X[i, :] + xk2*dt/2, *args)
        xk4 = f(t[i] + dt, X[i, :] + xk3*dt, *args)
        X[i + 1, :] = X[i, :] + (dt/6)*(xk1 + 2*xk2 + 2*xk3 + xk4)
        t[i + 1] = t[i] + dt

    # Adams-Bashforth-Moulton
    i = 3
    AB0 = f(t[i  ], X[i,   :], *args)
    AB1 = f(t[i-1], X[i-1, :], *args)
    AB2 = f(t[i-2], X[i-2, :], *args)
    AB3 = f(t[i-3], X[i-3, :], *args)

    for i in range(3,num_steps):
        #predico
        X[i + 1, :] = X[i, :] + dt/24*(55*AB0 - 59*AB1 + 37*AB2 - 9*AB3)
        t[i + 1] = t[i] + dt
        #correggo
        AB3 = AB2
        AB2 = AB1
        AB1 = AB0
        AB0 = f(t[i+1], X[i + 1, :], *args)

        X[i + 1, :] = X[i, :] + dt/24*(9*AB0 + 19*AB1 - 5*AB2 + AB3)

    return X, t

if __name__ == '__main__':

    #Parametri simulazione
    num_steps = 100000
    tf = 100
    #condizioni iniziali
    N = 1000
    I0 = 3
    R0 = 0
    D0 = 0
    S0 = N - I0 - R0 - D0
    y0 = [S0, I0, R0, D0]
    #parametri del modello
    beta   = 0.4
    gamma  = 0.03
    delta  = 0.01
    params = (beta, gamma, delta)
    #risolvo
    sol, t = AMB4(num_steps, tf, SIRD, y0, args=params)
    S, I, R, D = sol.T


    plt.figure(1)
    plt.title('Modello SIRD', fontsize=20)
    plt.plot(t, S, 'blue' , label='Suscettibili')
    plt.plot(t, I, 'red'  , label='Infetti')
    plt.plot(t, R, 'green', label='Rimessi')
    plt.plot(t, D, 'black', label='morti')
    plt.legend(loc='best')
    plt.grid()

    plt.show()