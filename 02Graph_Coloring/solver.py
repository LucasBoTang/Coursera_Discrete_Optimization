#!/usr/bin/python
# -*- coding: utf-8 -*-

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    global node_count, edge_count
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    # show the number of nodes and edges
    print('\n')
    print('Number of Nodes:', node_count)
    print('Number of Edges:', edge_count)

    # build a adjacency matrix of a graph
    import numpy as np
    global graph, node_degree
    graph = np.zeros((node_count, node_count), dtype='int8')
    node_degree = np.zeros(node_count, dtype='int16')

    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        graph[int(parts[0])][int(parts[1])] = 1
        graph[int(parts[1])][int(parts[0])] = 1
        node_degree[int(parts[0])] += 1
        node_degree[int(parts[1])] += 1

    # initialize the number of color and constraints
    cur_num = 0
    constraints = []
    for _ in range(node_count):
        constraints.append(set())
    solution = [None] * node_count
    visited = []

    # record the number of solutions and calculation
    global solution_count, calculation_count
    solution_count = 0
    calculation_count = 0

    # upper bound of chromatic number
    global min_num
    min_num = node_degree.max() + 1
    print('Upper Bound of Chromatic Number:', str(min_num))

    # Welsh-Powell algorithm
    print('Using Welsh-Powell algorithm...')
    wp(cur_num, solution, visited, constraints)

    # depth first search for constraint programming
    print('Using depth-first search...')
    import sys
    sys.setrecursionlimit(3000)
    dfs(cur_num, solution, visited, constraints)
    if solution_count < 2:
        print('No better solution within 1000 steps...')

    # randomly restart
    print('Using randomly restarts...')

    import random
    indexes = list(range(node_count))
    restart_count = 0

    for i in random.sample(indexes, min(100, node_count)):

        restart_count += 1
        calculation_count = 0
        if restart_count % 10 == 0:
            print('Randomly restart {} times...'.format(restart_count))

        # depth-first search
        new_solution, new_visited, new_constraints = propagation(i, 0, solution, visited, constraints)
        dfs(1, new_solution, new_visited, new_constraints)

    if solution_count < 3:
        print('No better solution within 1000 steps...')

    # prepare the solution in the specified output format
    output_data = str(min_num) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, best_solution))

    return output_data



def wp(cur_num, solution, visited, constraints):

    # sort by degree of nodes
    import numpy as np
    indexes = np.argsort(node_degree)[::-1]
    i = 0

    # get solution when all nodes are visited
    while not getSolution(visited, cur_num, solution):

        # use new color when no nodes can be colored
        if i == node_count:
            cur_num += 1
            i = 0
            continue

        # when the node is visited
        if indexes[i] in visited:
            i += 1
            continue

        else:
            # check the constraint
            if cur_num not in constraints[indexes[i]]:
                solution, visited, constraints = propagation(indexes[i], cur_num, solution, visited, constraints)
            i += 1


def dfs(cur_num, solution, visited, constraints):

    # check if get solution
    if getSolution(visited, cur_num, solution):
        return

    # sort by first fail principle
    indexes = firstFailSort(constraints, visited)

    for i in indexes:

        # limit times of calculation
        global calculation_count
        if calculation_count > 1000:
            return

        # get the color and update number of colors
        color, new_num = getColor(i, cur_num, constraints)

        # check the bound
        if new_num >= min_num:
            calculation_count += 1
            continue

        # propagate
        new_solution, new_visited, new_constraints = propagation(i, color, solution, visited, constraints)

        # recurse
        dfs(new_num, new_solution, new_visited, new_constraints)



def firstFailSort(constraints, visited):

    # the most number of constraints first, then highest degree
    helper = []
    for i in range(node_count):
        if i not in visited:
            helper.append((i, len(constraints[i]), node_degree[i]))
    helper.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # get the index
    return [x[0] for x in helper]



def getColor(i, cur_num, constraints):

    # symmetry break for color by choosing lowest color
    for color in range(cur_num+1):

        # check the constraint
        if color in constraints[i]:
            continue

        # when using a new color
        if color == cur_num:
            new_num = cur_num + 1
        # when not using a new color
        else:
            new_num = cur_num

        return color, new_num



def propagation(i, color, solution, visited, constraints):

    # deep copy
    new_solution = solution[:]
    new_visited = visited[:]
    new_constraints = [constraint.copy() for constraint in constraints]

    # color the node and propagation
    new_solution[i] = color
    new_visited.append(i)
    for j in range(node_count):
        if graph[i, j]:
            new_constraints[j].add(color)

    return new_solution, new_visited, new_constraints



def getSolution(visited, cur_num, solution):

    global min_num, best_solution, solution_count, calculation_count

    # get solution when all nodes are visited
    if len(visited) == node_count:

        # get one solution for the first time
        if not solution_count:
            min_num, best_solution = cur_num, solution
            print('Current Result:', str(min_num))

        # get a better solution
        if cur_num < min_num:
            min_num, best_solution = cur_num, solution
            print('Current Result:', str(min_num))

        solution_count += 1
        calculation_count += 1

        # get a final solution
        return True

    # not get the solution yet
    return False



if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
