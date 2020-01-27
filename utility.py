# Set of functions that can be used with Nurikabe

# isAdj checks two coordinates to see if theyre adjacent and returns a boolean


def isAdj(coord1, coord2):
    # For clarity
    x1, y1 = coord1
    x2, y2 = coord2
    # return (abs(x1-x2) <= 1 and abs(y1-y2) <= 1)
    return (abs(x2-x1) + abs(y2-y1) == 1)

# checks a list to see if any is adj


def isAdjinList(coordlist, coord):
    for coordinate in coordlist:
        if isAdj(coordinate, coord):
            return True
    return False

# returns the coordinate from coordlist1 that is adj to coordlist2 otherwise returns 0


def coordAdjbetweenTwoLists(coordlist1, coordlist2):
    for coord1 in coordlist1:
        for coord2 in coordlist2:
            if(isAdj(coord1, coord2)):
                return coord1
    return 0

# Checks to see if coordinate 1 is in range of coordinate 2, based on center
# value length


def inRange(centerValue, coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    distance = abs(x2-x1) + abs(y2-y1)
    if (distance <= centerValue):
        return True
    return False


def islandWorks(island, value):
    island_copy = island.copy()
    connectedIsland = []
    coordsAdjinclCenter = []
    searching = True

    coordsAdjinclCenter.append(island_copy.pop(0))
    while(searching):
        adjCoord = coordAdjbetweenTwoLists(island_copy, coordsAdjinclCenter)
        if (adjCoord != 0):
            coordsAdjinclCenter.append(
                island_copy.pop(island_copy.index(adjCoord)))
        else:
            searching = False

        connectedIsland = coordsAdjinclCenter

    return True if len(connectedIsland) == value else False
