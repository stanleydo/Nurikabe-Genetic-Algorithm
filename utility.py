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
    distance = abs(x1-x2) + abs(y1-y2)
    if (distance <= centerValue-1):
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

def findAllConnected(adjacents, starting_island, island_size, islands_in_range):

    if island_size == 1:
        return [[starting_island]]
    all_islands = []
    max_ttl = island_size

    fringe = []
    visited = []
    ttl = 0
    fringe.append((starting_island, visited.copy(), ttl))
    while fringe:
        cur_island, cur_visited, cur_ttl = fringe.pop(-1)

        # print("CUR TTL: ", cur_ttl)

        if cur_ttl == max_ttl:
            if set(cur_visited) not in all_islands:
                all_islands.append(set(cur_visited))
        else:
            if cur_ttl < max_ttl:
                if cur_island not in cur_visited:
                    cur_visited.append(cur_island)
                    cur_ttl += 1

                    filtered_adjacents = [coord for coord in adjacents[cur_island] if coord in islands_in_range]
                    for child in filtered_adjacents:

                        fringe.append((child, cur_visited.copy(), cur_ttl))

    return list(list(i) for i in all_islands)

def generateAdjacencies(coordinates, grid_size):
    adjacencies = dict()
    for coord in coordinates:
        adjacents = []
        x, y = coord
        adjacents += [(x + 1, y)] if x + 1 < grid_size else []
        adjacents += [(x, y + 1)] if y + 1 < grid_size else []
        adjacents += [(x - 1, y)] if x - 1 >= 0 else []
        adjacents += [(x, y - 1)] if y - 1 >= 0 else []
        adjacencies[coord] = adjacents
    return adjacencies

def avoidCoordinates(coordinates, adjacencies):
    definitely_avoid = dict()
    for coord in coordinates:
        avoid_all = [x for x in coordinates if x != coord]
        bad_adjacents = [j for i in [adjacencies[x] for x in avoid_all] for j in i]
        avoid_all = avoid_all + bad_adjacents
        definitely_avoid[coord] = avoid_all
    return definitely_avoid
