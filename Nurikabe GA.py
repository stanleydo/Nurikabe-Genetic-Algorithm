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
class NurikabeGA():

    # Grid size indicates a NxN grid
    grid_size = 5

    # Specifies the center island coordinates
    center_coords = [(0, 3), (2, 1), (2, 3), (4, 1)]

    # Private classes
    # Like A = Population()
    # Like B = Individual()

    gene_pool = [(x, y) for x, y in range(grid_size)
                 if (x, y) not in center_coords]


        # Maybe
    def mutate(self):
        pass

class Population():

    def __init__(self):
        pass

class Individual():

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
    # nurikabe = NurikabeGA()
    gene = Gene()
    gene.island = True
    print(gene.island)
    return 0

if __name__ == "__main__":
    main()