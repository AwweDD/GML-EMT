from EA_Fusion import Population
import numpy as np


class PopEvolution:
    def __init__(self, Dim):
        self.Dim = Dim

    def Evolve(self, Code, parents, CalObjs, w, t):
        Offspring = Population.population_(0, 0, 0, 0, 0)
        Par1 = parents[::2][:]
        Par2 = parents[1::2][:]
        mutation_rate = 1 / self.Dim
        if Code == 'real':
            if t < 1:
                # Simulated binary crossover
                proC = 1
                disC = 20
                proM = 1
                disM = 20
                beta = np.zeros((len(Par1), self.Dim))
                mu = np.random.rand(len(Par1), self.Dim)
                for i in range(len(beta)):
                    beta[i][np.where(mu[i] <= 0.5)[0]] = (2 * mu[i][np.where(mu[i] <= 0.5)[0]]) ** (1 / (disC + 1))
                    beta[i][np.where(mu[i] > 0.5)[0]] = (2 - 2 * mu[i][np.where(mu[i] > 0.5)[0]]) ** (-1 / (disC + 1))
                    beta[i] = beta[i] * -1
                    randReal = np.random.randint(0, 2, (1, self.Dim))
                    beta[i][np.where(randReal[0] == 0)[0]] = beta[i][np.where(randReal[0] == 0)[0]] * -1
                    beta[i][np.where(np.random.rand(1, self.Dim)[0] < 0.5)[0]] = 1
                    if np.random.rand() > proC:
                        beta[i] = np.ones((1, self.Dim))
                OffspringDec = np.concatenate(((Par1 + Par2) / 2 + beta * ((Par2 - Par1) / 2), (Par1 + Par2) / 2 - beta * ((Par2 - Par1) / 2)), axis=0)
                OffspringDec = np.minimum(np.maximum(OffspringDec, 0), 1)
                # Polynomial mutation
                Site = np.random.rand(CalObjs.N, self.Dim) < mutation_rate
                mu = np.random.rand(CalObjs.N, self.Dim)
                temp = np.bitwise_and(Site, mu <= 0.5)
                OffspringDec[temp] = OffspringDec[temp] + ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - OffspringDec[temp]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
                temp = np.bitwise_and(Site, mu > 0.5)
                OffspringDec[temp] = OffspringDec[temp] + (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - (1 - OffspringDec[temp])) ** (disM + 1)) ** (1 / (disM + 1)))
            else:
                # crossover
                OffspringDec = []
                for i in range(len(Par1)):
                    p1 = Par1[i][:]
                    p2 = Par2[i][:]
                    p_max = np.max((p1, p2), axis=0)
                    p_min = np.min((p1, p2), axis=0)
                    #
                    t = w * p_max + (1 - w) * p_min
                    #
                    u = np.random.rand(self.Dim)
                    beta = np.where(u <= 0.5, (2 * u) ** (1 / (20 + 1)), (1 / (2 * (1 - u))) ** (1 / (20 + 1)))
                    #
                    c1 = t - 0.5 * beta * (p2 - p1)
                    c2 = t + 0.5 * beta * (p2 - p1)
                    OffspringDec.append(c1)
                    OffspringDec.append(c2)
                OffspringDec = np.minimum(np.maximum(np.array(OffspringDec), 0), 1)
                # mutation
                alpha = 2 * (w - np.min(w)) / (np.max(w) - np.min(w)) - 1
                mu = np.random.rand(CalObjs.N, self.Dim)
                delta = np.where(mu < 0.5, ((2 * mu + (1 - 2 * mu) * (1 - OffspringDec) ** (20 + 1)) ** (1 / (20 + 1)) - 1),
                                 (1 - (2 * (1 - mu) + 2 * (mu - 0.5) * (1 - OffspringDec) ** (20 + 1)) ** (1 / (20 + 1))))
                delta_prime = alpha.reshape(1, -1) * np.abs(delta)
                #
                Site = np.random.rand(CalObjs.N, self.Dim) < mutation_rate
                OffspringDec[Site] = OffspringDec[Site] + delta_prime[Site]
        else:
            OffspringDec = []
            for i in range(len(Par1)):
                p1 = Par1[i][:]
                p2 = Par2[i][:]
                c1 = np.copy(p1)
                c2 = np.copy(p2)
                # crossover
                nonEq = np.where(p1 != p2)[0]
                c1[nonEq] = 1
                c2[nonEq] = 0
                c1[nonEq[w[nonEq] < np.random.rand(len(nonEq))]] = 0
                c2[nonEq[w[nonEq] > np.random.rand(len(nonEq))]] = 1
                # mutate
                mutation_pos = np.where(((np.random.rand(self.Dim) < mutation_rate) * 1 == 1))
                c1[mutation_pos] = 1 - c1[mutation_pos]
                c2[mutation_pos] = 1 - c2[mutation_pos]
                OffspringDec.append(c1)
                OffspringDec.append(c2)
            OffspringDec = np.array(OffspringDec)

        # Calculate the obj of Offspring
        PopObj, PopFit, Right_index = CalObjs.cal_objects_train(OffspringDec, Code)
        Offspring.Decs = OffspringDec
        Offspring.Objs = PopObj
        Offspring.Fits = PopFit
        Offspring.Coding = Code
        Offspring.Right_index = Right_index
        return Offspring
