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


class NurikabeGA():

    # Grid size indicates a NxN grid
    grid_size = 5

    # Specifies the center island coordinates
    center_coords = [(0, 3), (2, 1), (2, 3), (4, 1)]

    gene_pool = [(x, y) for x, y in range(grid_size)
                 if (x, y) not in center_coords]

    class Individual(list):

        # Initialize individual with a random set of (x,y) coordinates
        def __init__(self):
            pass

        def calculateFitness(self):
            pass

    class Population(list):

        def __init__(self):
            pass

        # Maybe
        def mutate(self):
            pass


def main():
    print("Hello World")
    nurikabe = NurikabeGA()
    print(nurikabe.gene_pool)
    print("test")
    return 0


if __name__ == "__main__":
    main()
