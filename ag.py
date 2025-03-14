from convert import Convert
import numpy as np
import matplotlib.pyplot as plt
import random


class AG:
    def __init__(self, min_x, max_x, f):

        self.convert_x = Convert(n_bits=4)
        self.population = []
        self.f = f
        self.min = min_x
        self.max = max_x

    def cost_fuction(self, chrom):
        x = self.convert_x.get_float(chrom)
        if 0 <= x <= np.pi:
            return self.f(x)
        else:
            return self.f(np.clip(x, a_min=self.min, a_max=self.max))

    def create_population(self, population_size, min, max):
        population = []
        for i in range(population_size):
            population.append(np.random.uniform(min, max))
        return population

    def crossover(self, father, mother):
        father = self.convert_x.get_bits(father)
        mother = self.convert_x.get_bits(mother)
        if len(father) == len(mother):
            crossover_point = random.randint(0, len(father) - 1)
            child1 = father[:crossover_point] + mother[crossover_point:]
            child2 = father[:crossover_point] + mother[crossover_point:]
            return child1, child2
        else:
            raise Exception("Father and Mother are not the same size")

    def mutate(self, x):
        bits = self.convert_x.get_bits(x)
        mutation_bit = np.random.uniform(0, len(bits) - 1)


    def new_population(self, fit_population, mutation, crossover):
        fx = np.array(fit_population)
        px = fx / np.sum(fx)
        new_pop = random.choices(population=fx, weights=px, k=len(fit_population))
        for i in range(1, len(new_pop), 2):
            father, mother = new_pop[i - 1], new_pop[i]
            if np.random.rand < crossover:
                child1, child2 = self.crossover(father, mother)
            else:
                child1, child2 = father, mother
            new_pop[i - 1] = child1
            new_pop[i] = child2
            if np.random.rand < mutation:
                mutated = self.mutate(father)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def function(x):
        return x + abs(np.sin(32 * x))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
