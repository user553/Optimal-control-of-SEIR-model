"""
SEIR disease model

S' = -beta * S * I
E' = beta * S * I  -  sigma * E
I' = sigma * E  -  nu * I
R' = nu * I
"""

import numpy as np
from ODESolver import ForwardEuler
from matplotlib import pyplot as plt

t_incubation = 5.1      #Average human incubation period  
t_infective = 3.3       #Average human infecive period
r = 2.4                 #Average no of people one person can infect the virus
N = 66000000            #population of the UK
E_initial = 1/N         
I_initial = 0.00        
R_initial = 0.00        
S_initial = 1 - E_initial - I_initial - R_initial

Sigma = 1/t_incubation
Nu = 1/t_infective
Beta = r * Nu

# contains the problem definition
class SEIR:
    def __init__(self, nu, beta, sigma, S0, E0, I0, R0):
        
        if isinstance(nu, (float, int)):
            self.nu = lambda t:nu
        elif callable(nu):
            self.nu = nu
        
        if isinstance(beta, (float, int)):
            self.beta = lambda t:beta
        elif callable(beta):
            self.beta = beta

        if isinstance(sigma, (float, int)):
            self.sigma = lambda t:sigma
        elif callable(sigma):
            self.sigma = sigma

        self.initial_conditions = [S0, E0, I0, R0]

    def __call__(self, u, t):
        S, E, I, R = u
        return np.asarray([
            -self.beta(t)*S*I, #susceptibles
            self.beta(t)*S*I - self.sigma(t)*E, #exposed
            self.sigma(t)*E - self.nu(t)*I, #infected
            self.nu(t)*I #recovered
        ])
        pass

#driver code for the SIR model
if __name__ == "__main__":
    seir = SEIR(Nu, Beta, Sigma, S_initial, E_initial, I_initial, R_initial)
    solver = ForwardEuler(seir)
    solver.set_initial_conditions(seir.initial_conditions)

    time_steps = np.linspace(0, 200, 101)
    u, t = solver.solve(time_steps)

    #plotting the results using matplotlib
    plt.plot(t, u[:, 0], label='Susceptible')
    plt.plot(t, u[:, 1], label='Exposed')
    plt.plot(t, u[:, 2], label='Infected')
    plt.plot(t, u[:, 3], label='Recovered')
    plt.ylabel("individuals")
    plt.xlabel("time")
    plt.legend()
    
    # plt.figure(figsize=(8,5))
    # plt.subplot(2,1,1)
    # plt.plot(t, u[:, 0], color='blue', lw=3, label='Susceptible')
    # plt.plot(t, u[:, 3], color='red', lw=3, label='Recovered')
    # plt.ylabel('Fraction')
    # plt.legend()

    # plt.figure(figsize=(8,5))
    # plt.subplot(2,1,1)
    # plt.plot(t, u[:, 1], color='orange', lw=3, label='Infected')
    # plt.plot(t, u[:, 2], color='purple', lw=3, label='Exposed')
    # plt.xlabel('Time(Days)')
    # plt.ylabel('Fraction')
    # plt.legend()

    plt.show()