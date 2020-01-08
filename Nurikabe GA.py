# http://www.logicgamesonline.com/nurikabe/
# 5x5 Nurikabe Genetic Algorithm
# Spencer Young
# Stanley Do

# Imports
import random
from operator import itemgetter
import time

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
# 1 6 1 1 1

best_individual =[(0,3),(0,0),(0,1),(0,2),(0,4),(2,1),(2,3),(2,4),(4,1),(3,0),(4,0),(4,2),(4,3),(4,4),(1,0),(1,1),(1,2),(1,3),(1,4),(2,0),(2,2),(3,1),(3,2),(3,3),(3,4)]

# Change breed to not maintain the top X elites, rather use a probability function to figure that out.
# Crossover should move an island from p1 into the child, then the rest into c1. Need to handle the duplicates.

# The class that combines everything to be called in main

# CONSTANTS
# Specify the grid size
# \/ \/ \/ \/ \/
grid_size = 5
# /\ /\ /\ /\ /\

list_size = grid_size * grid_size

# The main island coordinates (x,y): value
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
center_coords = {(0, 3): 5, (2, 1): 1, (2, 3): 2, (4, 1): 6}
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

# Dictionary which contains all of the valid edges for the corresponding coordinate
adjacencies = dict()
for coord in all_coords:
    adjacents = []
    x, y = coord
    adjacents += [(x+1, y)] if x+1 < grid_size else []
    adjacents += [(x, y+1)] if y+1 < grid_size else []
    adjacents += [(x-1, y)] if x-1 >= 0 else []
    adjacents += [(x, y-1)] if y-1 >= 0 else []
    adjacencies[coord] = adjacents

# TODOS .... Need to add multiple Populations and find a way to do multi-objective fitness.
class NurikabeGA():

    def __init__(self, grid_size, center_coords, generations):
        # Grid size indicates a NxN grid
        self.grid_size = grid_size

        # Specifies the center island coordinates
        self.center_coords = center_coords

        # Creates a list of all possible coordinates in a 5x5 grid
        self.gene_pool = [(x, y) for y in range(self.grid_size)
                          for x in range(self.grid_size) if (x, y) not in self.center_coords]

        self.generations = generations

    def geneticAlgorithm(self, pop_size, mating_pool_size, elite_size, mutation_rate, multi_objective_fitness=False):
        startTime = time.time()

        population = Population(
            pop_size=pop_size, mating_pool_size=mating_pool_size, elite_size=elite_size, mutation_rate=mutation_rate)

        best_individual = Individual()
        best_fitness = 0
        best_generation = 0

        for i in range(0, self.generations):
            for ind in population.population:
                # print("Individual: ", ind.individual)
                fitness = ind.calculate_fitness()
                if fitness > best_fitness:
                    best_individual = ind
                    best_fitness = fitness
                    best_generation = i

                if best_fitness == 4:
                    break

            if best_fitness == 4:
                print("Solution found in", round(
                    time.time() - startTime, 2), "seconds")
                print("Generation ", i, ": Best Fitness = ", best_fitness)
                print("Best Individual: ", best_individual.individual)
                best_individual.printAsMatrix()
                break

            if i % 10 == 0:
                print("Generation ", i, ": Best Fitness = ", best_fitness)
                print("Best Individual: ", best_individual.individual)

            population.breedPopulation()

    def mutate(self):
        pass


class Population():

    def __init__(self, pop_size, mating_pool_size, elite_size, mutation_rate, multi_objective_fitness=False):

        self.population = []

        self.pop_size = pop_size
        self.mating_pool_size = mating_pool_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

        for _ in range(pop_size):
            self.population.append(Individual(multi_objective_fitness))

    # How should we pick our mating pool?
    # I think Abbott likes having a little bit of tournament selection -
    # - Selecting a random sample (like 100), then evaluating fitness and sorting them from best to worst.
    def getMatingPool(self):
        mating_pool = []

        sample = random.sample(self.population, k=self.mating_pool_size)

        for individual in sample:
            mating_pool.append([individual, individual.calculate_fitness()])

        sorted_pool = sorted(mating_pool, key=lambda x: x[1], reverse=True)
        only_mates = [x for x, y in sorted_pool]

        return only_mates

    # The crossover (or replace an island part with an ocean)
    def breed(self, mating_pool):
        children = []

        # Maintain some elites based on elite size
        for i in range(self.elite_size):
            children.append(mating_pool[i])

        #
        for i in range(self.mating_pool_size - self.elite_size):
            random_parent1 = random.randint(0, self.mating_pool_size-1)
            random_parent2 = random.randint(0, self.mating_pool_size-1)

            parent1 = mating_pool[random_parent1]
            parent2 = mating_pool[random_parent2]

            matching_coords = [
                x for x in parent1.individual if x in parent2.individual and x not in center_coords_keys]

            # Randomly select two island coordinates
            try:
                random_coords = random.sample(matching_coords, k=2)

                p2_a_index = parent2.index(random_coords[0])
                p2_b_index = parent2.index(random_coords[1])

                child = parent2.copy()
                child[p2_a_index] = random_coords[1]
                child[p2_b_index] = random_coords[0]

                children.append(child)
            except:
                pass

            children.append(Individual())

        for i in range(self.pop_size - self.mating_pool_size):
            children.append(Individual())

        return children

    # Moves a randomly selected island from Parent 1 and puts it into parent 2.
    # Anything that was in parent 1 that's found in parent 2 will be replaced with the remainders.
    # Added ocean swapping as well.
    def single_island_crossover(self, mating_pool):
        children = []

        # Maintain some elites based on elite size
        for i in range(self.elite_size):
            children.append(mating_pool[i])

        for i in range(self.mating_pool_size - self.elite_size):
            # Find some random parents
            p1_random: Individual = random.choice(mating_pool)
            p2_random: Individual = random.choice(mating_pool)

            # random_island_range picks a random range like [0,5] or [5,10]
            p1_island_range = p1_random.random_island_range()

            # Finds the actual elements in the range and converts it to a set
            if len(p1_island_range) != 1:
                p1_toSet = set(
                    p1_random.individual[p1_island_range[0]+1:p1_island_range[1]])
                p2_toSet = set(
                    p2_random.individual[p1_island_range[0]+1:p1_island_range[1]])
                avoid_range = range(p1_island_range[0] + 1, p1_island_range[1])
            else:
                p1_toSet = set(
                    p1_random.individual[p1_island_range[0]:list_size])
                p2_toSet = set(
                    p2_random.individual[p1_island_range[0]:list_size])
                avoid_range = range(p1_island_range[0], list_size)

            # DEBUGGING
            # print("P1: ", p1_random.individual)
            # print("P2: ", p2_random.individual)

            # Maintain p1_toSet but as a list, so we can use it for easy comparison
            p1_toList = list(p1_toSet)
            # Subtracting the p1 set from the p2 set and making sure we don't lose any coordinates
            remainders = list(p2_toSet - p1_toSet)

            # DEBUGGING
            # print("p1_toList: ", p1_toList)
            # print("Remainders: ", remainders)

            child = Individual()
            child.individual = p2_random.individual.copy()
            inner = 0

            for i in range(list_size):
                if i in avoid_range:
                    child.individual[i] = p1_toList[inner]
                    inner += 1
                else:
                    if child.individual[i] in p1_toList:
                        child.individual[i] = remainders.pop()

            # Debugging
            # print("Parent 1 : ", p1_random.individual)
            # print("Parent 2 : ", p2_random.individual)
            # print("Child    : ", child.individual)

            children.append(child)

        for i in range(self.pop_size - self.mating_pool_size):
            children.append(Individual())

        return children

    # Mutates all of the individual in a given population
    # In this case, it should be all the children

    def mutate(self, children):
        for child in children:
            rand_chance = random.random()
            if rand_chance < self.mutation_rate:
                # print("Before Mutated Child: ", child.individual)
                random_land = random.randint(0, max_islands-1)
                while random_land in cum_sum_butlast:
                    random_land = random.randint(0, max_islands-1)
                random_ocean = random.randint(max_islands, list_size-1)
                temp_coord = child.individual[random_land]
                child.individual[random_land] = child.individual[random_ocean]
                child.individual[random_ocean] = temp_coord
                # print("After Mutated Child: ", child.individual)
            # This will mutate the individual into the solution
            # Use this for testing #
            # if rand_chance < 0.1:
            #     print("DING DING DING!")
            #     child.individual = best_individual
            #     print("BEST INDIVIDUAL FITNESS: ", child.calculate_fitness())
        return children

    def breedPopulation(self):
        mating_pool = self.getMatingPool()
        children = self.single_island_crossover(mating_pool)
        mutated = self.mutate(children)
        self.population = mutated
        return self

    # The tool to print an individual in a population as a matrix
    # 0 representing water, 1-X representing islands
    # Example:
    # Population #0:
    # [1, 0, 0, 1, 0]
    # [0, 2, 1, 0, 0]
    # [3, 1, 0, 3, 0]
    # [1, 0, 0, 0, 0]
    # [0, 0, 0, 0, 0]

    # Convert Individual to a matrix for visualization, Takes an index (int) of a population
    def printAsMatrix(self, index):
        # Initializing grid
        grid = [[0]*grid_size for i in range(grid_size)]

        # islandNumber is used to mark the grid
        islandNumber = 1
        # Using cumsum to get the indices of the individual to extract
        for i in range(len(cum_sum)-1):
            for x, y in self.population[index].individual[cum_sum[i]:cum_sum[i+1]]:
                # Assign the value in the grid
                grid[x][y] = islandNumber
            islandNumber += 1

        for _ in grid:
            print(_)


class Individual():

    # Structure of the individual is just this --> [(x,y), (x,y), ... (x,y)]
    # Randomly creates an individual with dedicated island indices - based off center_coords
    def __init__(self, multi_objective_fitness=False):

        self.individual = []

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
                self.individual.append(center_coords_keys[isl])
                isl += 1
            else:
                # Pop & append a coordinate from the newly created randomized list of valid coords to the individual.
                self.individual.append(random_valid_coords.pop())

        ocean_start_index = cum_sum[-1]
        self.ocean = self.individual[ocean_start_index:len(self.individual)]
        self.empty_list = [[0 for x in range(grid_size)] for y in range(grid_size)]

    # FITNESS FUNCTIONS SUBJECT TO CHANGE!!!
    # Just a regular fitness function

    def calculate_fitness(self):
        total_fitness = 0

        # isIsolated() will return a value indicating how many good (or isolated) islands there are.
        # A perfectly fit individual will have a fitness equal to the length.
        total_fitness += self.isIsolated()

        return total_fitness

    # focus_island should be an island index that we want to focus on
    # We can also specify a weight to the fitness
    # The additional args are for multi objective fitness
    def calculate_mo_fitness(self, focus_island, weight):
        total_fitness = 0

        # TODO

        return total_fitness

    # isAdj checks two coordinates to see if theyre adjacent and returns a boolean
    def isAdj(self, coord1, coord2):
    # For clarity
        x1,y1 = coord1
        x2,y2 = coord2
        return abs(((x2-x1)+(y2-y1))) == 1

    # Kinda tough
    # TODO
    def findConnected(self):
        # to know if a island is connected, the difference between each coordinate summed together must be |1|, example: (1,1)(1,2) = 1-1 + 1-2 = 1
        # using cum sum, we can calculate the number of connected islands
        # we can start by making a list of adjacent nodes

        # Currently a work in progress
        coordsAdj = []
        first = False
        numOfAdj = 0
        for i in range(len(cum_sum)-1):
            first = True
            for coord in self.individual[cum_sum[i]:cum_sum[i+1]]:
                # First marks center coordinate
                if(first):
                    first = False
                    center = coord
                else:
                    if(self.isAdj(center,(coord))):
                        numOfAdj += 1
                        #Add to list of coordinates that are adj to the center then redo
                        # have a rechecking counter
                        coordsAdj.append(coord)
                        # TODO restart procedure checking all coords in coordsadj using
                        # self.individual[cum_sum[i]:cum_sum[i+1]]:


            # print(numOfAdj)
            numOfAdj = 0
            first = True
        pass

    def isIsolated(self):
        # The island's adjacencies should only contain itself or an ocean.
        # Keeps track of iterations
        isl = 0

        # Incorporated a fitness value
        fitness_val = 0

        # For each coordinate in an island, the coordinate's adjacent position must be within itself or the ocean.
        for island_start in cum_sum_butlast:
            island_end = cum_sum[isl+1]

            # island is a list or splice of coordinates corresponding to an island
            island = self.individual[island_start:island_end]
            other_islands = list(set(self.individual[0:cum_sum[-1]])-set(island))

            # print("ISLAND: ", island)
            # print("OTHER ISLANDS: ", other_islands)

            # An island will stay "Good" if it's isolated.
            good_island = True

            # print("ISLAND: ", island)
            ## TODO ##
            # Extremely inefficient!! MAKE IT BETTER!
            all_adjacents = []
            if len(island) != 1:
                for coord in island:
                    adjacents = adjacencies[coord]
                    for a in adjacents:
                        all_adjacents.append(a)
                all_adjacents_no_dupes = list(set(all_adjacents))
                for a in all_adjacents_no_dupes:
                    if a in other_islands:
                        good_island = False
                for coord in island:
                    if coord not in all_adjacents_no_dupes:
                        good_island = False

            if good_island:
                fitness_val += 1

            isl += 1

        return fitness_val

    # Returns a list with each index corresponding to an island.
    # [-1, 1, -1] means the first island is not isolated, while the second one is. Third island is not isolated.
    def islandsIsolated(self):
        # The island's adjacencies should only be within itself or within an ocean.
        # Keeps track of iterations
        isl = 0

        # Incorporated a fitness value
        fitness_val = []

        # For each coordinate in an island, the coordinate's adjacent position must be within itself or the ocean.
        for island_start in cum_sum_butlast:
            island_end = cum_sum[isl+1]

            # island is a list or splice of coordinates corresponding to an island
            island = self.individual[island_start:island_end]

            # An island will stay "Good" if it's isolated.
            good_island = True

            for coord in island:
                adjacents = adjacencies[coord]
                for a in adjacents:
                    # Ocean is just a list/splice of the ocean. It's in the init of the individual.
                    if a not in self.ocean or a not in island:
                        good_island = False

            #
            if not good_island:
                fitness_val.append(-1)
            else:
                fitness_val.append(1)

            isl += 1

        return fitness_val

    # Returns a random island range
    # Also does oceans now
    def random_island_range(self):
        island_start_index = random.choice(range(len(cum_sum)))
        # print("Island Indices being swapped: ",
        #       cum_sum[island_start_index:island_start_index+2])
        return cum_sum[island_start_index:island_start_index + 2]

    def printAsMatrix(self):

        island_number = 1
        ct = 0
        for x,y in self.individual:
            if ct in cum_sum_butlast[1:]:
                island_number += 1
            elif ct >= cum_sum[-1]:
                island_number = 0

            self.empty_list[x][y] = island_number

            ct += 1

        for row in self.empty_list:
            print(row)


def main():
    # print("Hello World")
    # nurikabe = NurikabeGA(grid_size, center_coords)
    # print("Grid Size: ", nurikabe.grid_size)
    # print("Gene Pool: ", nurikabe.gene_pool)

    # Temporarily putting a bunch of print statements and instantiating some individuals & populations in main
    # Just for testing and debugging. Shouldn't be too hard to call it from NurikabeGA() later.

    # print("\n", "INFORMATION")
    # print("center_coords = dictionary of (coord): value --> ", center_coords)
    # print("valid_coords = list of non-center coordinates --> ", valid_coords)
    # print("cum_sum = list of INDICES where an island/ocean starts. Last element is the index where ocean starts. --> ", cum_sum, "\n")
    # print("cum_sum_butlast = ", cum_sum_butlast)

    # individual = Individual()
    # print("individual = ", individual)
    # individual.findConnected()
    # print("Individual Fitness: ", individual.calculate_fitness())

    # population = Population(
    #     pop_size=100, mating_pool_size=50, elite_size=10, mutation_rate=0.5)
    # print(population.getMatingPool())
    # print("Population = ", population)

    # The tool to print an individual in a population
    # print("Individual #0:")
    # population.printAsMatrix(0)
    

    nurikabe = NurikabeGA(grid_size=grid_size, center_coords=center_coords, generations=2000)
    nurikabe.geneticAlgorithm(
        pop_size=1000, mating_pool_size=800, elite_size=20, mutation_rate=0.5)

    # ind = Individual()
    # print(ind.individual)

    # print(ind)
    # print(ind.isAdj((3,4),(3,3)))
    # print(ind.findConnected())
    return 0


if __name__ == "__main__":
    main()
