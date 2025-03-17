import numpy as np
import matplotlib.pyplot as plt
import random
import struct


class Convert:
    def __init__(self, n_bytes):
        self.n_bits = n_bytes
        self.L = n_bytes * 8  # Total number of bits

        # Determine struct format based on bit size
        if self.L == 16:
            self.format_char = 'e'  # Float16
        elif self.L == 32:
            self.format_char = 'f'  # Float32
        else:
            raise ValueError("Unsupported bit size. Use 16-bit or 32-bit floats.")

    def floatToBits(self, f):
        """Convert float to bit representation as an integer."""
        s = struct.pack(f'>{self.format_char}', f)
        return int.from_bytes(s, byteorder='big')

    def bitsToFloat(self, b):
        """Convert integer bit representation back to float."""
        s = b.to_bytes(self.n_bits, byteorder='big')
        return struct.unpack(f'>{self.format_char}', s)[0]

    def get_bits(self, x):
        """Convert float to binary string representation."""
        x = self.floatToBits(x)
        return format(x, f'0{self.L}b')  # Convert to binary with leading zeros

    def get_float(self, bits):
        """Convert binary string back to float."""
        assert len(bits) == self.L, "Invalid bit string length"
        x = int(bits, 2)  # Convert binary string to integer
        return self.bitsToFloat(x)


class AG(Convert):
    def __init__(self, n_bytes, min_x, max_x, f, mutate_rate, crossover_rate):
        super().__init__(n_bytes)
        self.population = []
        self.f = f
        self.min = min_x
        self.max = max_x

        self.score = []
        if 0 <= mutate_rate > 1 or 0 <= crossover_rate > 1:
            raise Exception("Mutate rate and crossover_rate must be between 0 and 1.")
        self.mutate_rate = mutate_rate
        self.crossover_rate = crossover_rate

    def cost_fuction(self, population):
        fit_pop = []
        for x in population:
            if self.min <= x <= self.max:
                fit_pop.append(self.f(x))
            else:
                fit_pop.append(0)
                # fit_pop.append(self.f(1e-6))
        return fit_pop

    def create_population(self, population_size):
        population = []
        for i in range(population_size):
            population.append(np.random.uniform(self.min, self.max))
        return population

    def crossover(self, father, mother):
        father = self.get_bits(father)
        mother = self.get_bits(mother)
        if len(father) == len(mother):
            crossover_point = random.randint(0, len(father) - 1)
            child1 = father[:crossover_point] + mother[crossover_point:]
            child2 = mother[:crossover_point] + father[crossover_point:]
            return self.get_float(child1), self.get_float(child2)
        else:
            raise Exception("Father and Mother are not the same size")

    def mutate(self, chrom):
        bits = self.get_bits(chrom)
        mutation_bit = int(np.random.uniform(0, len(bits)-1))
        bits_list = list(bits)
        if bits_list[mutation_bit] == '0':
            bits_list[mutation_bit] = '1'
        else:
            bits_list[mutation_bit] = '0'
        mutated_chrom = ''.join(bits_list)
        mutated_chrom = self.get_float(mutated_chrom)
        return mutated_chrom

    def new_population(self,population, fit_population):
        self.score.append(np.mean(fit_population))
        father, mother = 0, 0
        new_population = []
        while len(new_population) < len(fit_population):
            father, mother = random.choices(population=population, weights=fit_population, k=2)

            if np.random.rand() < self.crossover_rate:
                child1, child2 = self.crossover(father, mother)
            else:
                child1, child2 = father, mother

            if np.random.rand() < self.mutate_rate:
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

            new_population.append(child1)
            new_population.append(child2)
        return new_population

    def plot_fit_curve(self):
        plt.figure()
        plt.plot(self.score)
        plt.legend(['Avg Fitness Score'])
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.show()

    def plot_fit_population(self, fit_population, population, epoch):
        plt.figure()
        x_f = np.linspace(start=0, stop=np.pi, num=500)
        y_f = self.f(x_f)
        plt.plot(x_f, y_f)
        plt.plot(population, fit_population, '.')
        plt.title('Fitness Curve, epoch {}'.format(epoch))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def function(x):
        return x + abs(np.sin(32 * x))

    AG = AG(n_bytes=4, min_x=0, max_x=np.pi, f=function, mutate_rate=0.003, crossover_rate=0.7)
    epochs = 600
    population = AG.create_population(population_size=100)
    for i in range(epochs):
        fit_population = AG.cost_fuction(population)
        new_population = AG.new_population(population, fit_population)
        if i%100 == 0:
            AG.plot_fit_population(fit_population=fit_population, population=population, epoch=i)
        population = new_population

    AG.plot_fit_curve()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
