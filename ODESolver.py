import numpy as np

# ODESolver superclass
class ODESolver:    

    # Constructor intialization that takes in the model defined in the other files. eg: SIR.py
    def __init__(self, f):
        self.f = f

    # Advance solution one time step, implemented in subclass below
    def advance(self):
        raise NotImplementedError

    # U0 are the initial conditons
    def set_initial_conditions(self, U0):

        # Scalar ODE, i.e; only one differential equation to solve
        if isinstance(U0, (int,float)):
            self.numbers_of_eqns = 1
            U0 = float(U0)

        # ODEs with more than one equation to solve or a system of equations
        else:
            U0 = np.asarray(U0)
            self.numbers_of_eqns = U0.size
        self.U0 = U0 #stored within the class

    # function to solve the equations using precision control using time_points
    def solve(self, time_points):

        self.t = np.asarray(time_points)
        n = self.t.size
        self.u = np.zeros((n, self.numbers_of_eqns))

        self.u[0, :] = self.U0 

        #integrating the equations
        for i in range(n -1):
            self.i = i
            self.u[i+1] = self.advance()

        return self.u[:i+2], self.t[:i+2]

#subclass
class ForwardEuler(ODESolver):
    def advance(self):
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i+1] - t[i]
        return u[i, :] + dt*f(u[i, :], t[i])