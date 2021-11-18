from math import log
from random import random

#class for SSA
class SSA:

    #Initialize container with model and pseudorandom number generator
    def __init__(self, model):
        
        self.model = model
        self.random = random

    #Indefinite generator of direct-method trajectories
    def direct(self):
        
        while True:
            while not self.model.exit():

                # evaluate weights and partition
                weights = [
                    (rxn, sto, pro(self.model))
                    for (rxn, sto, pro) in self.model.reactions
                ]
                partition = sum(w[-1] for w in weights)

                # evaluate sojourn time (MC step 1)
                sojourn = log(1.0 / self.random()) / partition
                self.model["time"].append(self.model["time"][-1] + sojourn)

                # evaluate the reaction (MC step 2)
                partition = partition * self.random()
                while partition >= 0.0:
                    rxn, sto, pro = weights.pop(0)
                    partition -= pro
                for species, delta in sto.items():
                    self.model[species].append(self.model[species][-1] + delta)

                self.model.curate()
            yield self.model
            self.model.reset()


    #Indefinite generator of 1st-reaction trajectories
    def first_reaction(self):
        
        while True:
            while not self.model.exit():

                # evaluate next reaction times
                times = [
                    (
                        log(
                            1.0 / self.random()
                        ) / pro(self.model),
                        sto
                    )
                    for (rxn, sto, pro) in self.model.reactions
                ]
                times.sort()

                # evaluate reaction time
                self.model["time"].append(
                    self.model["time"][-1] + times[0][0]
                )

                # evaluate reaction
                for species, delta in times[0][1].items():
                    self.model[species].append(
                        self.model[species][-1] + delta
                    )

                self.model.curate()
            yield self.model
            self.model.reset()