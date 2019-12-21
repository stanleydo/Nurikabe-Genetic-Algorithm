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

    # Maybe
    def mutate(self):
        pass


class Population(list):

    def __init__(self):
        pass


def main():
    print("Hello World")
    nurikabe = NurikabeGA()
    print(nurikabe.gene_pool)
    return 0


class Individual(list):

    # Initialize individual with a random set of (x,y) coordinates
    # Private Gene Class
    # Genes = Gene()

    def __init__(self):
        pass

    def calculateFitness(self):
        pass


if __name__ == "__main__":
    main()
