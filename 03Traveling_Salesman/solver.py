#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
from matplotlib import pyplot as plt
import random
import sys



def solve_it(input_data):

    # parse the input
    lines = input_data.split('\n')

    global node_count
    node_count = int(lines[0])

    global points
    points = np.zeros((node_count, 2))
    for i in range(0, node_count):
        line = lines[i+1]
        parts = line.split()
        points[i] = [float(parts[0]), float(parts[1])]

    # show the size of problem
    print('\n')
    print('Number of Nodes:', node_count)

    # build a distance matrix
    print('Build a distance matrix...')
    global distances
    distances = distanceMatirx(points)

    # generate an initial solution by greedy algorithm
    print('Starting with greedy algorithm...')
    solution = greedy()

    # calculate the length of the tour
    obj = pathLength(solution)

    # use simulated annealing alorithm with k-opt
    print('Using simulated annealing alorithm...')
    # set the intial temperature and cooling rate
    T0 = 1000
    alpha = 0.9
    solution, obj = simulatedAnnealing(T0, alpha, solution, obj, 'greedy')

    # reheat the simulated annealing
    print('Reheat...')

    # set the times of reheats
    if node_count < 500:
        M = 150
    elif node_count < 10000:
        M = 90
    else:
        M = 60

    for i in range(1, M+1):
        # choose nodes randomly
        node_choose = 'random'

        if i and i % 30 == 0:
            print('Reheat {} times...'.format(i))
            # chose node greedily every 10 times
            node_choose = 'greedy'

        solution, obj = simulatedAnnealing(T0, alpha, solution, obj, node_choose)

    # plot the path
    pathPlot(solution)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data



def distanceMatirx(points):

    # initialize the squared matrix
    distances = np.zeros((node_count, node_count))

    # calculate the distance by matrix multiplication
    if node_count <= 10000:
        G = np.dot(points, points.T)
        H = np.tile(np.diag(G), (node_count, 1))
        return np.sqrt(H + H.T - 2 * G)

    # avoid memory error
    else:
        for i in range(node_count-1):
            temp = np.array([points[i]] * node_count)
            i_distances = np.sqrt(np.square(temp[i:, 0] - points[i:, 0]) + np.square(temp[i:, 1] - points[i:, 1]))
            distances[i, i:] = i_distances
            distances[i:, i] = i_distances

            # display progress
            if i and i % 10000 == 0:
                print('Filled {} nodes...'.format(i))

        return distances



def pathPlot(solution):

    plt.ion()

    # sort as the solution
    co_ords = points[solution, :]
    # get x, y seperately
    x, y = co_ords[:, 0], co_ords[:, 1]
    plt.plot(x, y, color='b')
    plt.scatter(x, y, color='r')

    # connect the last node to the first node
    x, y = [co_ords[-1, 0], co_ords[0, 0]], [co_ords[-1, 1], co_ords[0, 1]]
    plt.plot(x, y, color='b')
    plt.scatter(x, y, color='r')

    plt.pause(15)
    plt.close()



def greedy():

    # start with the first node
    cur = 0
    solution = [0]

    # if all nodes are visisted
    while len(solution) != node_count:

        # visit the nearest unvisited node
        unvisited_distances = np.delete([distances[cur], range(node_count)], solution, axis=1)
        cur = int(unvisited_distances[:, np.argmin(unvisited_distances[0, :])][1])
        solution.append(cur)

    return solution



def pathLength(solution):

    obj = distances[solution[-1], solution[0]]
    for i in range(0, node_count-1):
        obj += distances[solution[i], solution[i+1]]

    return obj



def simulatedAnnealing(T, alpha, solution, obj, node_choose):

    best_solution, best_obj = solution, obj

    # store temperature and obj as list for visualization
    T_list = [T]
    obj_list = [obj]

    # set the number of steps
    if node_count < 10000:
        N = 1000
    else:
        N = 300

    # set upper bound of k
    k_upper = 100

    if node_choose == 'random':
        # randomly choose node
        nodes = [random.randrange(node_count) for _ in range(N)]

    elif node_choose == 'greedy':
        # choose the logest x1 edge first
        nodes = length_sort(solution)[:N]

    # set find flag
    find = False

    for node in nodes:

        # use k-opt for local search
        cur_solution, cur_obj = kOpt(solution, obj, node, k_upper)

        # when current solution is better
        if cur_obj <= obj:
            obj, solution = cur_obj, cur_solution[:]

        # when current solution is worse
        else:
            # calculate probability
            prob = 1 / (np.exp((cur_obj - obj) / T))
            # generate random number
            rand_num = random.random()
            # accept current solution with probability
            if rand_num <= prob:
                obj, solution = cur_obj, cur_solution

        # find a better solution
        if int(obj) < int(best_obj):
            find = True

        # store best solution and obj
        if obj <= best_obj:
            best_solution, best_obj = solution[:], obj

        # cool down
        T = alpha * T

        # append temperature and obj to list for visualization
        T_list.append(T)
        obj_list.append(obj)

    # visualize when find a better cur_solution
    if find:
        plt.ion()
        plt.xlabel("Temperature")
        plt.ylabel("Distance")
        plt.gca().invert_xaxis()
        plt.plot(T_list, obj_list)
        plt.pause(5)
        plt.close()

    return best_solution, best_obj



def length_sort(solution):

    # initialize list of solution length
    lengths = []

    # get the length between nodes
    for i in range(0, node_count-1):
        lengths.append([solution[i], distances[solution[i], solution[i+1]]])
    lengths.append([solution[-1], distances[solution[-1], solution[0]]])


    # sort nodes by length
    lengths.sort(key = lambda x: x[1], reverse=True)
    nodes = np.array(lengths, dtype=np.int)[:, 0]

    return nodes



def kOpt(solution, obj, t1, k_upper):

    # start with 2-opt
    k = 2

    # initialize solution and obj
    best_solution = []
    best_obj = float('inf')

    # stop when k achieve upper bound
    while k <= k_upper:

        temp_solution, temp_obj, end = kOptIterate(solution, obj, t1)

        # keep the best solution
        if temp_obj <= best_obj:
            best_solution, best_obj = temp_solution, temp_obj


        # stop when distance of x2 < distance of x1 do not exist
        if end:
            break

        # increase k
        k += 1

    return best_solution, best_obj



def kOptIterate(solution, obj, t1):

    # initialize end flag
    end = False

    # get t2
    t1_index = solution.index(t1)
    t2 = solution[(t1_index + 1) % node_count]

    # get edge x1 = (t1, t2)
    x1_length = distances[t1, t2]

    # get rank of length between t2 and others
    length_rank = np.argsort(distances[t2])
    # choose t3 correctly
    if solution[(t1_index + 2) % node_count] == np.argsort(distances[t2])[1]:
        length_rank = length_rank[2:]
    else:
        length_rank = length_rank[1:]

    if distances[t2, length_rank[0]] < x1_length:

        # get t3 randomly
        x2_length = float('inf')
        while x2_length >= x1_length:
            t3 = random.choice(length_rank[:25])
            # get x2 = (t2, t3)
            x2_length = distances[t2, t3]

        # get t3 greedily
#        t3 = length_rank[0]
#        x2_length = distances[t2, t3]

        # get t4
        t3_index = solution.index(t3)
        t4_index = (t3_index - 1) % node_count
        t4 = solution[t4_index]
        # get solution and obj
        temp_solution = swap(solution, obj, t1_index, t4_index)
        temp_obj = obj - distances[t1, t2] - distances[t3, t4] + distances[t1, t4] + distances[t2, t3]

    else:
        # stop k-opt
        end = True
        temp_solution, temp_obj = solution[:], obj

    return temp_solution, temp_obj, end



def swap(solution, obj, t1_index, t4_index):
    a, b = sorted([t1_index, t4_index])
    temp_solution = solution[:a+1] + solution[b:a:-1] + solution[b+1:]
    return temp_solution



if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
