# http://www.logicgamesonline.com/nurikabe/
# 5x5 Nurikabe Genetic Algorithm
# Spencer Young
# Stanley Do

# Imports
import random

# SAMPLE
# 0 0 0 5 0
# 0 0 0 0 0
# 0 1 0 2 0
# 0 0 0 0 0
# 0 6 0 0 0

# SOLUTION
# 1 1 1 5 1
# 0 0 0 0 0
# 0 1 0 2 1
# 1 0 0 0 0
# 1 1 1 1 1


# The class that combines everything to be called in main

# CONSTANTS
# Specify the grid size
# \/ \/ \/ \/ \/
grid_size = 5
# /\ /\ /\ /\ /\

list_size = grid_size * grid_size

# The main island coordinates (x,y): value
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
center_coords = {(0, 3): 5, (1, 1): 1, (2, 3): 2, (4, 1): 6}
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

center_coords_keys = list(center_coords.keys())
center_coords_vals = list(center_coords.values())

# Cumulative sum for indexing within an individual
# Every number but the last indicates where an island starts.
# The last number indicates where the ocean starts
s = 0
cum_sum = [0]
for x in center_coords_vals:
    s = s + x
    cum_sum.append(s)

# All the indices of where islands start.
cum_sum_butlast = cum_sum[:-1]

max_islands = s
max_waters = list_size - s

# A list that contains all of the (x,y) coordinates corresponding to the grid size.
all_coords = [(x, y) for y in range(grid_size) for x in range(grid_size)]

# A list that does not contain the center coords.
valid_coords = [x for x in all_coords if x not in center_coords]


class NurikabeGA():

    def __init__(self, grid_size, center_coords):
        # Grid size indicates a NxN grid
        self.grid_size = grid_size

        # Specifies the center island coordinates
        self.center_coords = center_coords

        # Creates a list of all possible coordinates in a 5x5 grid
        self.gene_pool = [(x, y) for y in range(self.grid_size)
                          for x in range(self.grid_size) if (x, y) not in self.center_coords]

    # Private classes

    # Like A = Population()

    # Maybe
    def mutate(self):
        pass


class Population(list):

    def __init__(self, pop_size, mating_pool_size, elite_size, mutation_rate, multi_objective_fitness=False):

        self.pop_size = pop_size
        self.mating_pool_size = mating_pool_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

        for _ in range(pop_size):
            self.append(Individual(multi_objective_fitness))

    # Not sure if we should evaluate the fitness of the individuals here, or do it when we initialize the population and store it in a list
    def evaluateIndividuals(self):
        pass

    # How should we pick our mating pool? TODO
    # I think Abbott likes having a little bit of tournament selection -
    # - Selecting a random sample (like 100), then evaluating fitness and sorting them from best to worst.
    def getMatingPool(self):
        mating_pool = []

        return mating_pool

    # TODO
    # The crossover (or replace an island part with an ocean)
    def breed(self, mating_pool):
        # For now just a random one?
        pass

    # TODO
    def mutate(self):
        pass

    # TODO
    def breedPopulation(self):
        pass

    # The tool to print an individual in a population 
    # Convert Individual to a matrix (Visualization), Takes an index (int) of a population
    def printAsMatrix(self, index):
        
        # Initializing grid
        grid = [[0]*grid_size for i in range(grid_size)]

        # islandNumber is used to mark the grid
        islandNumber = 1
        # Using cumsum to get the indices of the individual to extract
        for i in range(len(cum_sum_butlast)-1):
            for x,y in self[index][cum_sum[i]:cum_sum_butlast[i+1]]:
                # Assign the value in the grid
                grid[x][y] = islandNumber
            islandNumber += 1
        
        # 0 is water, 1-X is islands numbers
        for _ in grid:
            print(_)


class Individual(list):

    # Structure of the individual is just this --> [(x,y), (x,y), ... (x,y)]
    # Randomly creates an individual with dedicated island indices - based off center_coords
    def __init__(self, multi_objective_fitness=False):

        self.multi_objective_fitness = multi_objective_fitness

        # ind keeps track of the index of the cum_sum_butlast
        isl = 0

        # Copy the list of valid coords and shuffle them in place
        random_valid_coords = valid_coords.copy()
        random.shuffle(random_valid_coords)

        # List building: Loop through size of the list ... (i = 0..24 for a 5x5)
        for i in range(list_size):

            # isl is initialized outside of the loop b/c it keeps track of the index of cum_sum_butlast
            # cum_sum_butlast is a list of indices where an island is supposed to start.
            if isl < len(cum_sum_butlast) and i == cum_sum_butlast[isl]:
                self.append(center_coords_keys[isl])
                isl += 1
            else:
                # Pop & append a coordinate from the newly created randomized list of valid coords to the individual.
                self.append(random_valid_coords.pop())

    # FITNESS FUNCTIONS SUBJECT TO CHANGE!!!
    # Just a regular fitness function

    def calculate_fitness(self):
        total_fitness = 0

        # TODO

        return total_fitness

    # focus_island should be an island index that we want to focus on
    # We can also specify a weight to the fitness
    # The additional args are for multi objective fitness
    def calculate_mo_fitness(self, focus_island, weight):
        total_fitness = 0

        # TODO

        return total_fitness

    # TODO
    # Kinda tough
    def findConnected(self):
        pass

    # TODO, island should be isolated: not connected to another island.
    def isIsolated(self):
        pass


# Not sure if we need a Gene() class
class Gene():
    coordinate = tuple
    island = bool
    centerValue = int
    connectedislands = int

    def __init__(self):
        pass
        # Coordinates (Set first index of associated island as the pre-defined center value/island)
        # Island or Ocean
        # Center Value
        # Total connected island or ocean


def main():
    # print("Hello World")
    # nurikabe = NurikabeGA(grid_size, center_coords)
    # print("Grid Size: ", nurikabe.grid_size)
    # print("Gene Pool: ", nurikabe.gene_pool)

    # Temporarily putting a bunch of print statements and instantiating some individuals & populations in main
    # Just for testing and debugging. Shouldn't be too hard to call it from NurikabeGA() later.

    print("\n", "INFORMATION")
    print("center_coords = dictionary of (coord): value --> ", center_coords)
    print("valid_coords = list of non-center coordinates --> ", valid_coords)
    print("cum_sum = list of INDICES where an island/ocean starts. Last element is the index where ocean starts. --> ", cum_sum, "\n")
    print("cum_sum_butlast = ", cum_sum_butlast)

    individual = Individual()
    print("individual = ", individual)

    population = Population(
        pop_size=100, mating_pool_size=100, elite_size=10, mutation_rate=0.5)
    print("Population = ", population)
    
    # The tool to print an individual in a population 
    print("Population #0:")
    population.printAsMatrix(0)

    return 0


if __name__ == "__main__":
    main()
