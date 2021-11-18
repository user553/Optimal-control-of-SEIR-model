from matplotlib import pyplot
from numpy import linspace
from scipy.integrate import odeint
from SSA import *
from SSAModel import *

# initial species counts and sojourn times
initital_conditions = {
    "s": [480],
    "i": [20],
    "r": [0],
    "time": [0.0],
}

# propensity functions
propensities = {
    0: lambda d: 2.0 * d["s"][-1] * d["i"][-1] / 500,
    1: lambda d: 1.0 * d["i"][-1],
}

# change in species for each propensity
stoichiometry = {
    0: {"s": -1, "i": 1, "r": 0},
    1: {"s": 0, "i": -1, "r": 1},
}

# instantiate the epidemic SSA model container
epidemic = SSAModel(
    initital_conditions,
    propensities,
    stoichiometry
)

# instantiate the SSA container with model
epidemic_generator = SSA(epidemic)


pyplot.figure(figsize=(20,20), dpi=500)

# make the plots
axes = pyplot.subplot(312)
axes.set_ylabel("individuals")
axes.set_xlabel("time (arbitrary units)")

# simulate and plot 30 trajectories
trajectories = 0
for trajectory in epidemic_generator.direct():
    axes.plot(trajectory["time"], trajectory["s"], color="orange")
    axes.plot(trajectory["time"], trajectory["i"], color="orange")
    axes.plot(trajectory["time"], trajectory["r"], color="orange")
    trajectories += 1
    if trajectories == 300:
        break

# numerical solution using an ordinary differential equation solver
t = linspace(0, 14, num=200)
y0 = (480, 20, 0)
alpha = 2.0
beta = 1.0

def differential_SIR(n_SIR, t, alpha, beta):
    dS_dt = -alpha * n_SIR[0] * n_SIR[1] / 500
    dI_dt = ((alpha * n_SIR[0] / 500) - beta) * n_SIR[1]
    dR_dt = beta * n_SIR[1]
    return dS_dt, dI_dt, dR_dt

solution = odeint(differential_SIR, y0, t, args=(alpha, beta))
solution = [[row[i] for row in solution] for i in range(3)]

# plot numerical solution
axes.plot(t, solution[0], color="black")
axes.plot(t, solution[1], color="black")
axes.plot(t, solution[2], color="black")

pyplot.show()