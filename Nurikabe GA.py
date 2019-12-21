# http://www.logicgamesonline.com/nurikabe/
# 5x5 Nurikabe Genetic Algorithm
# Spencer Young
# Stanley Do

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

# Constants
grid_size = 5
center_coords = [(0, 3), (2, 1), (2, 3), (4, 1)]


class NurikabeGA():
    grid_size = int
    center_coords = list
    gene_pool = list

<<<<<<< HEAD
    # Grid size indicates a NxN grid
    grid_size = 5
=======
    def __init__(self, grid_size, center_coords):
        # Grid size indicates a NxN grid
        self.grid_size = grid_size
>>>>>>> 1798b2f7ec16d606751b630122883a8a5acff672

        # Specifies the center island coordinates
        self.center_coords = center_coords

        # Creates a list of all possible coordinates in a 5x5 grid
        self.gene_pool = [(x, y) for y in range(self.grid_size)
                          for x in range(self.grid_size) if (x, y) not in self.center_coords]

    # Private classes
    # Like A = Population()
    # Like B = Individual()

<<<<<<< HEAD
    gene_pool = [(x, y) for x, y in range(grid_size)
                 if (x, y) not in center_coords]


        # Maybe
=======
    # Maybe
>>>>>>> 1798b2f7ec16d606751b630122883a8a5acff672
    def mutate(self):
        pass


class Population(list):

    def __init__(self):
        pass


class Individual(list):

    # Initialize individual with a random set of (x,y) coordinates
    # Private Gene Class
    # Genes = Gene()

    def __init__(self):
        pass

    def calculateFitness(self):
        pass


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
<<<<<<< HEAD
    # nurikabe = NurikabeGA()
=======
    print("Hello World")
    nurikabe = NurikabeGA(grid_size, center_coords)
    print("Gene Pool: ", nurikabe.gene_pool)

>>>>>>> 1798b2f7ec16d606751b630122883a8a5acff672
    gene = Gene()
    gene.island = True
    print(gene.island)
    return 0


if __name__ == "__main__":
    main()