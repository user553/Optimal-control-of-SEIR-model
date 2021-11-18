"""
SEIRS disease model

S' = nu*N - beta*S*I + omega*R - nu*S
E' = beta*S*I - sigma*E - nu*E
I' = sigma*E - gamma*I - (nu + alpha)*I
R' = gamma*I - omega*R - nu*R
"""

import numpy as np
from ODESolver import ForwardEuler
from matplotlib import pyplot as plt

t_incubation = 5.1      #Average human incubation period  
t_infective = 3.3       #Average human infecive period
r = 2.4                 #Average no of people one person can infect the virus
Age_exp = 81*365        #Avegrage age expentency in the UK
N = 660000000           #Population of the UK
E_initial = 1/N         
I_initial = 0.00        
R_initial = 0.00        
S_initial = 1 - E_initial - I_initial - R_initial

Sigma = 1/t_incubation
Gamma = 1/t_infective
Beta = r * Gamma
Alpha = 0
Nu_death = 1/Age_exp
Nu_birth = 0
Omega = 1/365

#contains the problem definition
class SEIRS:
    def __init__(self, nu_birth, nu_death, beta, sigma, gamma, alpha, omega, S0, E0, I0, R0):
        
        if isinstance(nu_birth, (float, int)):
            self.nu_birth = lambda t:nu_birth
        elif callable(nu_birth):
            self.nu_birth = nu_birth

        if isinstance(nu_death, (float, int)):
            self.nu_death = lambda t:nu_death
        elif callable(nu_death):
            self.nu_death = nu_death
        
        if isinstance(beta, (float, int)):
            self.beta = lambda t:beta
        elif callable(beta):
            self.beta = beta

        if isinstance(sigma, (float, int)):
            self.sigma = lambda t:sigma
        elif callable(sigma):
            self.sigma = sigma

        if isinstance(gamma, (float, int)):
            self.gamma = lambda t:gamma
        elif callable(gamma):
            self.gamma = gamma

        if isinstance(alpha, (float, int)):
            self.alpha = lambda t:alpha
        elif callable(alpha):
            self.alpha = alpha

        if isinstance(omega, (float, int)):
            self.omega = lambda t:omega
        elif callable(omega):
            self.omega = omega

        self.initial_conditions = [S0, E0, I0, R0]


    def __call__(self, u, t):
        S, E, I, R = u
        return np.asarray([
            self.nu_birth(t)*N - self.beta(t)*S*I + self.omega(t)*R - self.nu_death(t)*S, #susceptibles
            self.beta(t)*S*I - self.sigma(t)*E - self.nu_birth(t)*E, #exposed
            self.sigma(t)*E - self.gamma(t)*I - (self.nu_birth(t) + self.alpha(t))*I, #infected
            self.gamma(t)*I - self.omega(t)*R - self.nu_birth(t)*R #recovered
        ])
        pass

#driver code for the problem
if __name__ == "__main__":
    seirs = SEIRS(Nu_birth, Nu_death, Beta, Sigma, Gamma, Alpha, Omega, S_initial, E_initial, I_initial, R_initial)
    solver = ForwardEuler(seirs)
    solver.set_initial_conditions(seirs.initial_conditions)

    time_steps = np.linspace(0, 300, 101)
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