import random
from operator import itemgetter
import time
from numpy.random import choice
import itertools
import math
import sys
from nurikabe.utility import *
import PySimpleGUI as sg
from random import randint

# CONSTANTS
# Specify the grid size
# \/ \/ \/ \/ \/
# grid_size = 5
# grid_size = 6
# grid_size = 7
# grid_size = 8
# grid_size = 10
grid_size = 15
# /\ /\ /\ /\ /\

list_size = grid_size * grid_size

MAX_ROWS = MAX_COL = grid_size
# board = [[randint(0,1) for j in range(MAX_COL)] for i in range(MAX_ROWS)]

layout =  [[sg.Text('', size=(4, 2), key=str((i,j)), pad=(1,1), text_color='white', background_color='white', justification='center') for j in range(MAX_COL)] for i in range(MAX_ROWS)] +\
          [[sg.Button('Exit', key='Exit')]] +\
          [[sg.Button('Run', key='Run')]] +\
          [[sg.Text('Generation: '), sg.Text('0', key='Generation', size=(10,0))]]

window = sg.Window('Nurikabe Genetic Algorithm', layout).Finalize()

# The main island coordinates (x,y): value
# GRID SIZE 5
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
# center_coords = {(0, 3): 5, (2, 1): 1, (2, 3): 2, (4, 1): 6}
# center_coords = {(0,2):2, (1,0):3, (3,4):2, (4,2):3}
# center_coords = {(0,0):1, (2,0):7, (3,3):1}
# center_coords = {(1,4):4, (3,1):1, (3,3):1}
# center_coords = {(0,0):5, (0,2):1, (0,4):3, (4,0):1, (4,2):1, (4,4):1}
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

# GRID SIZE 6
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
# center_coords = {(1,1):1, (2,0):5, (2,2):3, (4,2):2, (4,5):6}
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

# GRID SIZE 7
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
# center_coords = {(1,2): 3, (1,4): 4, (2,1): 1, (2,5): 1, (3,3): 1, (4,1): 4, (4,5): 1, (5,2): 1, (5,4): 4}
# center_coords = {(0,1):6, (0,3):2, (2,6):5, (3,5):6, (5,5):1}
# center_coords = {(0,0):19}
# center_coords = {(0,0):1, (0,4):1, (0,6):7, (2,0):4, (4,6):1, (6,0):5, (6,2):3, (6,6):1}
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

# GRID SIZE 8
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
# center_coords = {(0,2):2, (0,6):3, (1,1):1, (2,3):1, (2,7):1, (3,1):2, (4,3):3, (5,4):4, (6,0):2, (6,7):3, (7,5):4}
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

# GRID SIZE 10
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
# center_coords = {(0,4):2, (0,8):3, (1,5):4, (2,4):5, (4,1):1, (4,4):2, (5,5):2, (5,8):2, (7,5):5, (8,4):4, (9,1):4, (9,5):2}
# center_coords = {(0,2):2, (0,5):1, (0,9):1, (1,1):2, (1,7):1, (2,3):7, (3,6):2, (6,6):1, (7,3):5, (7,8):6, (8,1):5, (8,7):1, (9,2):2, (9,5):1, (9,9):1}
# Broken? \/
# center_coords = {(0,3):1, (0,6):2, (0,9):1, (1,1):2, (1,7):1, (2,4):1, (2,6):1, (7,3):5, (7,5):7, (8,2):7, (8,8):7, (9,3):2, (9,6):2}
# center_coords = {(0,0):3, (0,2):2, (0,4):6, (0,7):8, (3,2):2, (3,8):1, (4,1):1, (5,8):2, (6,1):3, (9,2):1, (9,5):5, (9,7):1, (9,9):11}
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

# Grid size 15
center_coords = {(0,9):2, (1,1):6, (1,13):3, (1,11):3, (2,10):4, (3,1):1, (3,5):1, (3,7):2, (4,2):4, (4,8):2, (5,0):7,  (7,11):1, (6,4):2, (6,6):2, (6,8):2, (7,3):2, (7,11):2, (8,6):2, \
                 (8,8):2, (8,10):3, (9,3):4, (9,14):7, (10,6):3, (10,12):3, (11,7):3, (11,9):4, (11,13):2, (12,4):2, (13,1):1, (13,3):1, (13,13):2, (14,5):1, (14,9):2}

center_coords_keys = list(center_coords.keys())
center_coords_vals = list(center_coords.values())
max_swaps = max(center_coords_vals) - 1

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
cum_sum_withmax = cum_sum.copy()
cum_sum_withmax.append(list_size)

max_islands = s
max_waters = list_size - s

# A list that contains all of the (x,y) coordinates corresponding to the grid size.
all_coords = [(x, y) for y in range(grid_size) for x in range(grid_size)]

# A list that does not contain the center coords.
valid_coords = [x for x in all_coords if x not in center_coords]

# Dictionary which contains all of the valid edges for the corresponding coordinate
adjacencies = generateAdjacencies(all_coords, grid_size)

# Create a list of coordinates to definitely avoid
# This means anything adjacent to any other main island, or the main islands themselves.
definitely_avoid = avoidCoordinates(center_coords_keys, adjacencies)

all_combinations = dict()

for coord in center_coords:
    all_coords_in_range = [a for a in all_coords if inRange(center_coords[coord], coord, a) and a not in definitely_avoid[coord]]
    possible_islands = []
    all_combinations[coord] = findAllConnected(adjacencies, coord, center_coords[coord], all_coords_in_range)

# print([x for x in all_combinations[(0,7)] if (2,9) in x])

class NurikabeGA():

    def __init__(self, grid_size, center_coords, generations, print_interval):

        # Grid size indicates a NxN grid
        self.grid_size = grid_size

        # Specifies the center island coordinates
        self.center_coords = center_coords

        # Number of generations the individuals may evolve
        self.generations = generations

        # Prints per generation
        self.print_interval = print_interval

    def geneticAlgorithm(self, pop_size, mating_pool_size=None, elite_size=None, mutation_rate=0.5):

        startTime = time.time()

        population = Population(pop_size, mating_pool_size, elite_size, mutation_rate)

        best_individual = Individual()
        best_fitness = 0
        best_generation = 0
        avg_fitness = 0

        # Top Loop
        while True:
            (event, value) = window.Read()
            if event == 'Exit':
                break
            elif event == 'Run':
                for i in range(0, self.generations):
                    cur_best_fit = 0
                    for ind in population.population:
                        fitness = ind.calculate_fitness()
                        avg_fitness += fitness
                        if fitness > cur_best_fit:
                            best_individual.individual = ind.individual.copy()
                            cur_best_fit = fitness
                            best_fitness = fitness
                            best_generation = i

                    if best_individual.isSolved():
                        print()
                        print("Nurikabe Solved in", time.time() - startTime, "seconds!")
                        print("Generation ", i, ": Best Fitness = ", best_fitness)
                        print("Best Individual: ", best_individual.individual)
                        print("Connected Islands: ", best_individual.findConnected())
                        print("Connected Oceans: ", best_individual.findConnectedOcean())
                        best_individual.printAsMatrix()
                        j = 0
                        next_coord = 0
                        for coord in best_individual.individual:
                            if j < max_islands:
                                if j in cum_sum_butlast[1:]:
                                    next_coord += 1
                                window[str(coord)].update(background_color='black', value=str(center_coords_vals[next_coord])+'\n#'+str(next_coord))
                            else:
                                window[str(coord)].update(background_color='white')
                            window['Generation'].update(value=i)
                            j += 1
                        window.Refresh()
                        break

                    if i % self.print_interval == 0:
                        print()
                        # print("Average Fitness: ", avg_fitness/pop_size)
                        # print("Generation ", i, ": Best Fitness = ", best_fitness)
                        # print("Best Individual: ", best_individual.individual)
                        # print("Connected Islands: ", best_individual.findConnected())
                        # print("Connected Oceans: ", best_individual.findConnectedOcean())
                        # print("# Islands Isolated: ", best_individual.isIsolated())
                        # print("Islands Not Isolated: ", best_individual.islandsNotIsolated())
                        # best_individual.fixRange()
                        # best_individual.printAsMatrix()
                        j = 0
                        next_coord = 0
                        for coord in best_individual.individual:
                            if j < max_islands:
                                if j in cum_sum_butlast[1:]:
                                    next_coord += 1
                                window[str(coord)].update(background_color='black', value=str(center_coords_vals[next_coord])+'\n#'+str(next_coord))
                            else:
                                window[str(coord)].update(background_color='white')
                            window['Generation'].update(value=i)
                            j += 1
                        window.Refresh()

                    avg_fitness = 0

                    population.breedPopulation()

    def breedPopulations(self, populations):
        random.shuffle(populations)
        for x in range(len(populations)-1):
            cur_pop = populations[x]
            next_pop = populations[x+1]

            ind_num = 0
            for ind in populations[x].population:
                breed_chance = random.random()
                if breed_chance < 0.5:
                    ind = populations[x+1].population[ind_num]
                else:
                    populations[x].population[ind_num] = ind
                ind_num += 1



class Population():

    def __init__(self, pop_size, mating_pool_size=None, elite_size=None, mutation_rate=0.5):

        # A population is a list of individuals
        self.population = []

        self.pop_size = pop_size
        self.mating_pool_size = mating_pool_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

        for _ in range(pop_size):
            self.population.append(Individual())

    # Moves a randomly selected island from Parent 1 and puts it into parent 2.
    # Anything that was in parent 1 that's found in parent 2 will be replaced with the remainders.
    # Added ocean swapping as well.
    def single_island_crossover(self, mating_pool):
        children = []
        elites = []

        # Maintain some elites based on elite size
        for i in range(self.elite_size):
            children.append(mating_pool[i])
            elites.append(mating_pool[i])

        for i in range(self.mating_pool_size - self.elite_size):
            # Find some random parents
            p1_random: Individual = random.choice(elites)
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

            children.append(child)

        for i in range(self.pop_size - self.mating_pool_size):
            children.append(Individual())

        return children

    # Mutates all of the individual in a given population
    # In this case, it should be all the children
    # Randomly swaps a child with an ocean
    def mutate(self, children):
        for child in children:

            rand_chance = random.random()

            solved_child = None
            #
            if rand_chance < self.mutation_rate:
                child.random_swap()
                if child.isSolved():
                    solved_child = child
            else:
                child.small_swap()
                if child.isSolved():
                    solved_child = child

            child.reduce_max_conflict()

            if child.isSolved():
                solved_child = child

            if solved_child:
                children[0] = solved_child

        return children

    def breedPopulation(self):
        # mating_pool = self.getMatingPool()
        # print("MP: ", mating_pool)
        # children = self.single_island_crossover(mating_pool)
        # # children = self.population.copy()
        # print("CHILDREN: ", children)
        mutated = self.mutate(self.population)
        # self.population = mutated
        return self


class Individual():

    def __init__(self):

        self.states = []

        self.individual = []
        self.islands_lists = []

        self.ocean = []
        self.islands_dict = dict()
        self.ocean_start_index = cum_sum[-1]
        self.all_isl_conflicts = dict()
        # ind keeps track of the index of the cum_sum_butlast
        isl = 0

        # Create a list of possible squares. from 1 to -2, 2 to -1 for x and y
        # This makes searching for ocean squares less costly
        self.squares = []
        for i in range(grid_size-1):
            for j in range(grid_size-1):
                self.squares.append(((i,j),(i+1,j),(i,j+1),(i+1,j+1)))

        # Copy the list of valid coords and shuffle them in place
        random_valid_coords = valid_coords.copy()
        random.shuffle(random_valid_coords)

        satisfied = False
        while len(self.individual) != list_size and not satisfied:
            self.individual = []
            for coord in center_coords:
                random_valid_island = random.choice(all_combinations[coord])
                self.individual = self.individual + random_valid_island
                self.islands_dict[coord] = random_valid_island
            self.ocean = []
            for coord in all_coords:
                if coord not in self.individual:
                    self.ocean.append(coord)
                    self.individual.append(coord)
            if len(self.findConnectedOcean()) == max_waters:
                satisfied = True

        for coord in center_coords:
            all_combs_for_coord = all_combinations[coord]

        for main_coord in center_coords:
            self.all_isl_conflicts[main_coord] = self.island_conflicts(main_coord)

        self.empty_list = [[0 for x in range(grid_size)] for y in range(grid_size)]

    def numConflicts_general(self):
        conflicts = 0
        conflicts += self.numOceanSquares()
        conflicts += len(self.islandsNotIsolated())
        conflicts += max_waters - len(self.findConnectedOcean())
        return conflicts

    def reduce_max_conflict(self):
        if self.isSolved():
            pass
        else:
            conflicts = [(coord, self.island_conflicts(coord)) for coord in center_coords]
            most_conflicted_island = random.choices(population=[x[0] for x in conflicts], weights=[y[1] for y in conflicts], k=1)[0]
            print("CONFLICTS: ", conflicts)
            print("Most conflicted: ", most_conflicted_island)
            if self.island_conflicts(most_conflicted_island) != 0:
                self.try_reducing(most_conflicted_island)
            else:
                self.random_swap()

    def random_swap(self, coord=None, small=False):
        large_coords = [(x,self.island_conflicts(x)) for x in center_coords if center_coords[x] >= 3]

        if not small:
            coord = random.choices(population=[x[0] for x in large_coords], weights=[x[1] for x in large_coords], k=1)[0]
        fitness_snapshot = self.calculate_fitness()
        individual_snapshot = self.individual.copy()
        best_individual_so_far = self.individual.copy()

        coord_index = self.individual.index(coord)
        size = center_coords[coord]

        island_range = [coord_index, coord_index + size]

        other_islands = self.individual[0:coord_index] + self.individual[coord_index + size:cum_sum[-1]]
        other_islands_adjacents = set(
            [x for sublist in [adjacencies[y] for y in other_islands] for x in sublist]).union(set(other_islands))

        all_individuals = []

        for working_island_segment in all_combinations[coord]:
            valid = True
            for working_coord in working_island_segment:
                if working_coord in other_islands_adjacents:
                    valid = False

            if valid:
                current_island = self.individual[coord_index:coord_index + size]
                same_coords = set(current_island).intersection(set(working_island_segment))
                coords_cur_but_work = list(set(current_island) - set(working_island_segment))
                coords_work_but_cur = list(set(working_island_segment) - set(current_island))
                replace_coords = []

                if not set(current_island) == same_coords:

                    j = 0
                    for i in range(island_range[0],island_range[1]):
                        if self.individual[i] in same_coords:
                            pass
                        else:
                            replace_coords.append(self.individual[i])
                            self.individual[i] = coords_work_but_cur[j]
                            j += 1

                    for i in set(range(list_size)) - set(range(island_range[0],island_range[1])):
                        if self.individual[i] in coords_work_but_cur:
                            self.individual[i] = coords_cur_but_work.pop()

                    if self.calculate_fitness() >= fitness_snapshot:
                        best_individual_so_far = self.individual.copy()
                        fitness_snapshot = self.calculate_fitness()
                        all_individuals.append((self.individual.copy(), fitness_snapshot))
                    else:
                        all_individuals.append((self.individual.copy(), self.calculate_fitness()))
                    self.individual = individual_snapshot
        if len(all_individuals) != 0:
            good_individual = random.choices(population=[x[0] for x in all_individuals], weights=[x[1] for x in all_individuals], k=1)[0]
            self.individual = good_individual
        else:
            self.individual = best_individual_so_far

    def small_swap(self):
        max_island_size = max(center_coords_vals)
        coord = random.choices(population=center_coords_keys, weights=[abs(1-((x-1)/max_island_size)**(1/2)) if x > 2 else 1for x in center_coords_vals], k=1)[0]
        self.random_swap(coord, True)

    def force_swap(self, coord):
        working_island_segment = random.choice(all_combinations[coord])

        # random_island_range picks a random range like [0,5] or [5,10]
        coord_index = self.individual.index(coord)
        size = center_coords[coord]
        island_range = [coord_index, coord_index + size]
        current_island = self.individual[coord_index:coord_index + size].copy()
        rest_of_current_island = [x for x in self.individual if x not in current_island]

        same_coords = set(current_island).intersection(set(working_island_segment))
        coords_cur_but_work = list(set(current_island) - set(working_island_segment))
        coords_work_but_cur = list(set(working_island_segment) - set(current_island))
        replace_coords = []

        j = 0
        for i in range(island_range[0], island_range[1]):
            if self.individual[i] in same_coords:
                pass
            else:
                replace_coords.append(self.individual[i])
                self.individual[i] = coords_work_but_cur[j]
                j += 1

        for i in set(range(list_size)) - set(range(island_range[0], island_range[1])):
            if self.individual[i] in coords_work_but_cur:
                self.individual[i] = coords_cur_but_work.pop()

    def reduce_random(self):
        coord = random.choice(center_coords_keys)
        self.try_reducing(coord)

    def try_reducing(self, coord, small_swap=False):
        best_ind_so_far = self.individual.copy()
        fitness_snapshot = self.calculate_fitness()
        conflicts_snapshot = self.island_conflicts(coord)
        validOnce = False
        for working_island_segment in all_combinations[coord]:
            # random_island_range picks a random range like [0,5] or [5,10]
            coord_index = self.individual.index(coord)
            size = center_coords[coord]
            island_range = [coord_index, coord_index+size]
            current_island = self.individual[coord_index:coord_index+size]

            same_coords = set(current_island).intersection(set(working_island_segment))
            coords_cur_but_work = list(set(current_island) - set(working_island_segment))
            coords_work_but_cur = list(set(working_island_segment) - set(current_island))

            other_islands = self.individual[0:coord_index] + self.individual[coord_index+size:cum_sum[-1]]
            other_islands_adjacents = set([x for sublist in [adjacencies[y] for y in other_islands] for x in sublist]).union(set(other_islands))

            if coord == (0,7):
                print("WORKING ISLAND SEGMENT: ", working_island_segment)

            valid = True
            for working_coord in working_island_segment:
                if working_coord in other_islands_adjacents:
                    valid = False
            if valid:
                validOnce = True
                j = 0
                for i in range(island_range[0], island_range[1]):
                    if self.individual[i] in same_coords:
                        pass
                    else:
                        self.individual[i] = coords_work_but_cur[j]
                        j += 1

                for i in set(range(list_size)) - set(range(island_range[0], island_range[1])):
                    if self.individual[i] in coords_work_but_cur:
                        self.individual[i] = coords_cur_but_work.pop()

                if self.isSolved():
                    self.solution = self.individual
                    best_ind_so_far = self.individual.copy()
                    break
                else:
                    conflicts = self.island_conflicts(coord)
                    if (self.calculate_fitness() >= fitness_snapshot and conflicts <= conflicts_snapshot):
                        fitness_snapshot = self.calculate_fitness()
                        best_ind_so_far = self.individual.copy()
                        conflicts_snapshot = self.island_conflicts(coord)
        self.individual = best_ind_so_far

    def calculate_fitness(self, island_focus=-1):
        total_fitness = 0.0

        # isIsolated() will return a value indicating how many good (or isolated) islands there are.
        # A perfectly fit individual will have a fitness equal to the length.

        oceans_fitness = self.connectedFitnessOcean()
        isolation_fitness = self.isIsolated()

        if oceans_fitness == max_waters:
            total_fitness += max_waters
        else:
            return oceans_fitness

        if self.isOceanSquare():
            return oceans_fitness - self.numOceanSquares()

        total_fitness += isolation_fitness
        if isolation_fitness == len(center_coords):
            pass
        else:
            return total_fitness

        if island_focus != -1:
            total_fitness += self.connectedFitness()
        else:
            total_fitness += self.connectedFitness()

        return total_fitness

    # This puts everything in seperate lists. i could have put this in find connected, but
    # this is easier to understand
    # this will return as many lists as there are islands
    def prepareIslandLists(self):
        tempList = []
        combinedLists = []
        for i in range(len(cum_sum)-1):
            for coord in self.individual[cum_sum[i]:cum_sum[i+1]]:
                tempList.append(coord)
            combinedLists.append(tempList)
            tempList = []
        return combinedLists

    # Returns a list of each connected island
    # Eg: 4 islands will return something like [[(0,3),(1,3)],[(2,1)],[(2,3),(3,3)],[(4,1)]]
    def findConnected(self):
        # Preparing the list of islands for calculation
        islands = self.prepareIslandLists()
        # Initializing a list of connected Island coordinates
        connectedIslands = []
        # Initializing the list that will be used to check for adjacencies
        coordsAdjinclCenter = []
        # This Boolean will specify when to stop searching for adjacencies (when no match is found)
        searching = True

        for island in islands:
            # Add the center and remove it from the island
            coordsAdjinclCenter.append(island.pop(0))
            while(searching):
                # Compare coordsAdjinclCenter with island. If any adj is found, add it
                # to coordsAdjinclCenter and remove it from the island

                # Temporary variable to reduce cost
                adjCoord = coordAdjbetweenTwoLists(island,coordsAdjinclCenter)
                # If no adj is found, coordAdjbetweenTwoLists will return 0
                if(adjCoord != 0):
                    coordsAdjinclCenter.append(island.pop(island.index(adjCoord)))
                else:
                    # No matches found, stop the search
                    searching = False

            connectedIslands.append(coordsAdjinclCenter)
            coordsAdjinclCenter = []
            searching = True
        return connectedIslands

    # Same as findConnected() but specifically for oceans
    def findConnectedOcean(self):
        ocean = self.individual[self.ocean_start_index:len(self.individual)]
        # Initializing a list of connected Island coordinates
        connectedOceans = []
        # Initializing the list that will be used to check for adjacencies
        coordsAdjinclCenter = []
        # This Boolean will specify when to stop searching for adjacencies (when no match is found)
        searching = True

        coordsAdjinclCenter.append(ocean.pop(0))
        while(searching):
            adjCoord = coordAdjbetweenTwoLists(ocean,coordsAdjinclCenter)
            # print("adJCOORD ", adjCoord)
            if(adjCoord != 0):
                coordsAdjinclCenter.append(ocean.pop(ocean.index(adjCoord)))
            else:
                # No matches found, stop the search
                searching = False

        connectedOceans.append(coordsAdjinclCenter)
        return connectedOceans

    def allConnectedOceans(self):

        ocean_unchanged = self.individual[self.ocean_start_index:len(self.individual)]

        # The set of ocean coordinates
        ocean = set(ocean_unchanged.copy())

        ocean_nodes_visited = set()

        all_connected_oceans = []

        for ocean_node in ocean:
            if ocean_node not in ocean_nodes_visited:
                fringe = [ocean_node]
                visited = set()
                while fringe:
                    current_ocean = fringe.pop(0)
                    ocean_nodes_visited.add(current_ocean)
                    visited.add(current_ocean)
                    ocean_adjacents = [x for x in adjacencies[current_ocean] if x in ocean_unchanged and x not in ocean_nodes_visited]
                    for adjacent in ocean_adjacents:
                        fringe.append(adjacent)
                all_connected_oceans.append(list(visited))
            ocean = ocean - ocean_nodes_visited

        return all_connected_oceans

    # TODO: check for the longest length ocean, not just the ones adj to the first
    def findConnectedOceans2(self):
        searching = True
        coordsAdjtoFirst = []
        ocean = self.individual[cum_sum[-1]:]
        coordsAdjtoFirst.append(ocean.pop(0))
        while(searching):
            adjCoord = coordAdjbetweenTwoLists(ocean,coordsAdjtoFirst)
            if(adjCoord != 0):
                coordsAdjtoFirst.append(ocean.pop(ocean.index(adjCoord)))
            else:
                searching = False
        return coordsAdjtoFirst

    def connectedOceanFitness2(self):
        bestOceanSize = list_size - cum_sum[-1]
        connectedOcean = self.findConnectedOceans2()
        if(len(connectedOcean) == bestOceanSize):
            # double the points if its the right size
            return bestOceanSize * 2
        return len(connectedOcean)

    # Returns whether or not there is a square in the ocean
    def isOceanSquare(self):
        for square in self.squares:
            if(set(square).issubset(self.individual[cum_sum[-1]:])):
                return True
        return False

    def findFirstOceanSquare(self):
        for square in self.squares:
            if(set(square).issubset(self.individual[cum_sum[-1]:])):
                return square
        return 0

    # Finds the number of squares in an ocean
    def numOceanSquares(self):
        ct = 0
        for square in self.squares:
            if(set(square).issubset(self.individual[cum_sum[-1]:])):
                ct += 1
        return ct

    # Finds and returns the set of ocean coordinates that create a square.
    def setOfOceanSquares(self):
        sqrs = set()
        # Squares in self.squares is a 4-size list of coordinates (sets)
        for square in self.squares:
            if set(square).issubset(self.individual[cum_sum[-1]:]):
                for sq_coord in square:
                    sqrs.add(sq_coord)
        return sqrs

    def smallestConnectedOceans(self):
        connected_oceans = sorted(self.allConnectedOceans(), key=len)[:-1]
        # print("CONNECTED OCEANS: ", connected_oceans)
        # print("OTHER THING: ", [y for x in connected_oceans for y in x])
        return [y for x in connected_oceans for y in x]

    # Given a coordinate returns whether or not it is an island
    def isIsland(self,coord):
        if(self.individual.index(coord) < cum_sum[-1]):
            return True
        return False

    def connectedFitness(self):
        connectedIslands = self.findConnected()
        # give a big fitness bonus if the size of the island is the correct size
        # first we have to identify the size each island must be
        # Using list comprehension, I can use zip to combine the same list to subtract
        # each value next to each other to give me the island sizes
        bestIslandSizes = [x2 - x1 for (x1, x2) in zip(cum_sum[0:], cum_sum[1:])]
        bestScore = max(bestIslandSizes)

        # the sum of the list created from: if the island is > than the size, -1 point.
        # if it is an incorrect size, then just add the the size of the island
        # if it is a correct size then add the size of the island with a bonus 3. (larger correct islands give bigger points)

        # Original Code Below
        # connectedFitness = sum([-1 if len(cIsland) > sizes else len(cIsland) if len(cIsland) != sizes
        # else len(cIsland)+3 for (cIsland, sizes) in zip(connectedIslands, bestIslandSizes)])

        # Changed for some testing
        connectedFitness = sum([-1 if len(cIsland) > sizes else len(cIsland) if len(cIsland) != sizes
        else bestScore for (cIsland, sizes) in zip(connectedIslands, bestIslandSizes)])

        return connectedFitness

    def connectedFitnessOcean(self):
        connectedOceans = self.findConnectedOcean()[0]

        # A different connectedFitness for testing
        connectedFitness = max_waters if len(connectedOceans) == max_waters else len(connectedOceans)

        # Update: i realized that a bonus 3 would mean that small islands with correct size would not be valuable
        # so instead, full size islands will get a max fitness, which is the highest island cost
        # connectedFitness = sum([-1 if len(cIsland) > sizes else len(cIsland) if len(cIsland) != sizes
        # else bestScore for (cIsland, sizes) in zip(connectedOceans, bestIslandSizes)])
        return connectedFitness

    # This is the main isIsolated function that is currently in use.
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

            # An island will stay "Good" if it's isolated.
            good_island = True

            ## TODO ##
            # Extremely inefficient!! MAKE IT BETTER!
            all_adjacents = []
            for coord in island:
                adjacents = adjacencies[coord]
                for a in adjacents:
                    all_adjacents.append(a)
            all_adjacents_no_dupes = set(all_adjacents)
            for coord in island:
                if coord not in all_adjacents_no_dupes and len(island) != 1:
                    good_island = False
            for a in all_adjacents_no_dupes:
                if a in other_islands:
                    good_island = False

            if good_island:
                fitness_val += 1

            isl += 1

        return fitness_val

    def island_conflicts(self, main_coord):
        # The island's adjacencies should only contain itself or an ocean.
        # Keeps track of iterations
        isl = 0

        # Incorporated a fitness value
        conflicts = 0

        ocean_squares = self.setOfOceanSquares()

        small_connected_oceans = self.smallestConnectedOceans()

        # For each coordinate in an island, the coordinate's adjacent position must be within itself or the ocean.
        for island_start in cum_sum_butlast:
            island_end = cum_sum[isl+1]

            if self.individual[island_start] == main_coord:
                # island is a list or splice of coordinates corresponding to an island
                island = self.individual[island_start:island_end]
                other_islands = list(set(self.individual[0:max_islands]) - set(self.individual[island_start:island_end]))
                other_islands_adjacents = set([x for sublist in [adjacencies[y] for y in other_islands] for x in sublist])

                ## TODO ##
                # Extremely inefficient!! MAKE IT BETTER!
                all_adjacents = []
                for coord in island:
                    adjacents = adjacencies[coord]
                    for a in adjacents:
                        all_adjacents.append(a)
                all_adjacents_no_dupes = set(all_adjacents)
                for coord in island:
                    if coord not in all_adjacents_no_dupes and len(island) != 1 or coord in other_islands_adjacents:
                        conflicts += 1
                        pass
                for a in all_adjacents_no_dupes:
                    if a in other_islands:
                        conflicts += 1
                    if a in ocean_squares:
                        conflicts += 1
                    if a in small_connected_oceans:
                        conflicts += 1
            else:
                pass
            isl += 1

        return conflicts

    # Returns a list with each index corresponding to an island.
    # [-1, 1, -1] means the first island is not isolated, while the second one is. Third island is not isolated.
    def islandsNotIsolated(self):
        # The island's adjacencies should only contain itself or an ocean.
        # Keeps track of iterations
        isl = 0

        # Incorporated a fitness value
        isolated_islands = []

        # For each coordinate in an island, the coordinate's adjacent position must be within itself or the ocean.
        for island_start in cum_sum_butlast:
            island_end = cum_sum[isl+1]

            # island is a list or splice of coordinates corresponding to an island
            island = self.individual[island_start:island_end]
            other_islands = list(set(self.individual[0:cum_sum[-1]])-set(island))

            # An island will stay "Good" if it's isolated.
            good_island = True

            ## TODO ##
            # Extremely inefficient!! MAKE IT BETTER!
            all_adjacents = []
            for coord in island:
                adjacents = adjacencies[coord]
                for a in adjacents:
                    all_adjacents.append(a)
            all_adjacents_no_dupes = set(all_adjacents)
            for coord in island:
                if coord not in all_adjacents_no_dupes and len(island) != 1:
                    good_island = False
            for a in all_adjacents_no_dupes:
                if a in other_islands:
                    good_island = False

            if good_island:
                pass
            else:
                isolated_islands.append(island)

            isl += 1

        return isolated_islands

    # Returns a random island range
    # Also does oceans now
    def random_island_range(self):
        island_start_index = random.choice(range(len(cum_sum_withmax)))
        # print("Island Indices being swapped: ",
        #       cum_sum[island_start_index:island_start_index+2])
        return cum_sum_withmax[island_start_index:island_start_index + 2]

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

    def isSolved(self):
        bestIslandSizes = [x2 - x1 for (x1, x2) in zip(cum_sum[0:], cum_sum[1:])]
        bestScore = max(bestIslandSizes)
        largest_fitness_possible = len(center_coords) + max_waters + (bestScore * len(center_coords))
        if self.isIsolated() == len(center_coords) and self.connectedFitnessOcean() == max_waters and not self.isOceanSquare() and self.calculate_fitness() == largest_fitness_possible:
            return True

    def allInRange(self):
        for i in range(len(cum_sum_butlast)):
            center = self.individual[cum_sum[i]]
            max_size = center_coords[center]
            for coord in self.individual[cum_sum[i]+1:cum_sum[i+1]]:
                if max_size != 1:
                    if inRange(max_size, center, coord):
                        pass
                    else:
                        return False
        return True

    def all_possible_islands(self, coord, value):
        if value == 1:
            return [set(coord)]

        return all_combinations[coord]

    def isAllPossible(self):
        i = 0
        score = 0
        for _ in cum_sum_butlast:
            if self.individual[cum_sum[i]:cum_sum[i+1]] not in self.all_possible_islands(self.individual[cum_sum[i]],center_coords[self.individual[cum_sum[i]]]):
                return score
            else:
                i += 1
                score += 1

        return score

    def curState(self):
        curstate = []
        for i in range(len(cum_sum_butlast)):
            island = []
            for k in range(cum_sum[i], cum_sum[i+1]):
                island.append(self.individual[k])
            curstate.append(set(island))
        return curstate

def main():
    nurikabe = NurikabeGA(grid_size=grid_size, center_coords=center_coords, generations=100000, print_interval=1)
    nurikabe.geneticAlgorithm(
        pop_size=1, mating_pool_size=2, elite_size=1, mutation_rate=0.65)
    return 0

if __name__ == "__main__":
    main()
