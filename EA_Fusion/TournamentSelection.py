import numpy as np


class ESelection:
    def __init__(self, N):
        self.N = N

    def TournamentSelection(self, fitness):
        parents = np.random.randint(0, self.N, (2, self.N))
        sel_parents = [parents[0][i] if fitness[parents[0][i]] > fitness[parents[1][i]] else parents[1][i] for i in range(parents.shape[1])]
        return np.array(sel_parents)
