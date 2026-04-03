from EA_Fusion import Population
import numpy as np


class PopulationGeneration:
    def __init__(self, Dim, task_num):
        self.Dim = Dim
        self.Task_num = task_num

    def generation(self, CalObjs):
        Populations = []
        Population1 = Population.population_(0, 0, 0, 0, 0)
        Population2 = Population.population_(0, 0, 0, 0, 0)
        """
            Task 1: initialization individuals
        """
        PopDec1 = np.random.uniform(low=0, high=1, size=(CalObjs.N, self.Dim))
        """
            Task 2: initialization individuals
        """
        PopDec2 = np.random.randint(2, size=(CalObjs.N, self.Dim))
        #
        Population1.Decs = PopDec1
        Population2.Decs = PopDec2
        Population1.Coding = 'real'
        Population2.Coding = 'binary'
        Populations.append(Population1)
        Populations.append(Population2)
        for i in range(self.Task_num):
            PopObjs, PopFits, Right_index = CalObjs.cal_objects_train(Populations[i].Decs, Populations[i].Coding)
            Populations[i].Objs = PopObjs
            Populations[i].Fits = PopFits
            Populations[i].Right_index = Right_index
        return Populations
