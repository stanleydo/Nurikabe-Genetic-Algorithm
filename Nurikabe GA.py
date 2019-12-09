import random
#http://www.logicgamesonline.com/nurikabe/
# Given:
# 0 0 0 0 0 
# 0 0 0 1 0
# 3 0 0 0 4
# 0 1 0 0 0
# 0 0 0 0 0

# Solution:
#  - - - - - 
#| 0 0 0 0 0 |
#| 1 1 0 1 0 |
#| 1 0 0 0 1 |
#| 0 1 0 1 1 |
#| 0 0 0 0 1 |
#  - - - - - 


# My world will have 0 for ocean, 1 for water, 9 for walls
class NurikabeGA():
    # Class initializations
    def __init__(self, coordinates):
        self.population = []
        self.generatePopulation()
        self.addIslands(coordinates)
        self.numberOfIslands = len(coordinates)/2
        self.fitnessScores = []
        self.generation = 1
        self.offspring = []

    
    # These Functions are for visualizations
    # Generating an empty world (no islands)
    def generatePopulation(self):
        board = []
        numContainer = [9,0,0,0,0,0,9]
        board.append([9,9,9,9,9,9,9])
        for i in range(5):
            board.append(numContainer.copy())
        board.append([9,9,9,9,9,9,9])
        self.population = board

    # Print Function
    def showPopulation(self):
        for i in self.population:
            print(i)

    # Adding starting islands
    def addIslands(self,coordinates):
        islandValues = [] 
        positions = []
        # Adding islands is in the format of (Value, tuple(XYCordinates))
        # Saving the values 
        for i in coordinates[0::2]:
            islandValues.append(i)
        for i in coordinates[1::2]:
            positions.append(i)

        # Now to add the new values and positions to the current world
        for i in range(len(islandValues)):
            self.population[positions[i][0]][positions[i][1]] = islandValues[i]
    
    # The all the islands are an individual would have the coordinates of all the islands
    # as suggested by you
    # def islandsIndividual(self, coordinates):
        # positions = self.numberOfIslands
    def checkForWalls(self, coordinates):
        # return a list strings of the direction of a walls, otheriwse return "None" 
        walls = []
        for x,y in coordinates:
            #left
            if self.population[x-1][y] == 9:
                walls.append("West")
            if self.population[x+1][y] == 9:
                walls.append("East")
            if self.population[x][y+1] == 9:
                walls.append("North")
            if self.population[x][y-1] == 9:
                walls.append("South")
            if len(walls) == 0:
                walls.append("None")
            return walls
            



    # trying island being an individual
class IndiviualIsland():
    def __init__(self, coordinates, size):
        # the core positions
        corePos = [coordinates]
        # list of positions
        positions = []
        islandSize = size 
        



        def mutate(self):
            if self.islandSize > 1:





        
    # class IndiviualOcean():








                




# add all adjacent 


def main():
    islands = [3, (3,1), 1, (2,4), 4, (3,5), 1, (4,2)]
    a = NurikabeGA(islands)
    a.showPopulation()
    # a.islandIndividual(1)
    
    
if __name__ == "__main__":
    main()
    