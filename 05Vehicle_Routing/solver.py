#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import numpy as np
import scipy.spatial.distance as ssd
from matplotlib import pyplot as plt


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    global customer_count, vehicle_count, vehicle_capacity
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    demands = np.empty(customer_count)
    x_coordinations = np.empty(customer_count)
    y_coordinations = np.empty(customer_count)
    for i in range(customer_count):
        line = lines[i+1]
        parts = line.split()
        demands[i] = int(parts[0])
        x_coordinations[i] = float(parts[1])
        y_coordinations[i] = float(parts[2])

    # show the size of problem
    print('\n')
    print('Number of customers:', customer_count)
    print('Number of vehicles:', vehicle_count)

    # build distance matirx
    print('Building distance matirx...')
    distances = distanceMatirx(x_coordinations, y_coordinations)

    # start with greedy algorithm
    obj, vehicle_tours, chromesome = greedy(demands, distances)

    # visualize solution
    visualize(vehicle_tours, x_coordinations, y_coordinations)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(vehicle_count):
        outputData += ' '.join([str(customer) for customer in vehicle_tours[v]]) + '\n'

    return outputData



def distanceMatirx(x_coordinations, y_coordinations):

    locations = np.column_stack((x_coordinations, y_coordinations))
    distances = ssd.cdist(locations, locations, 'euclidean')

    return np.array(distances)



def greedy(demands, distances):

    # initialize solution
    obj = 0
    vehicle_tours = []
    chromesome = []

    # start with depot
    cur_customer = 0
    vehicle_index = 1
    cur_capacity = vehicle_capacity
    vehicle_tour = []
    used = set([0])

    # if all customers are visisted
    while len(used) < customer_count:
        # find the closest feasible unvisted customer
        sorted_customers = np.argsort(distances[cur_customer])
        i = 0

        while i < customer_count:
            # check availability
            if i in used:
                i += 1
                continue
            # check feasibility
            if cur_capacity == 0:
                break
            if demands[i] > cur_capacity:
                i += 1
                continue

            # update a tour
            obj += distances[cur_customer, i]
            chromesome.append(i)

            cur_customer = i
            cur_capacity -= demands[i]
            vehicle_tour.append(i)
            used.add(i)

        # reset when one tour is completed
        vehicle_tours.append([0]+vehicle_tour+[0])
        cur_customer = 0
        vehicle_index += 1
        cur_capacity = vehicle_capacity
        vehicle_tour = []

    vehicle_index -= 1
    # penalize if infeasible
    if vehicle_index > vehicle_count:
        obj = float('inf')
    # format for output
    else:
        for _ in range(vehicle_count - vehicle_index):
            vehicle_tours.append([0, 0])

    return obj, vehicle_tours, chromesome



def visualize(vehicle_tours, x_coordinations, y_coordinations):

    plt.ion()
    plt.figure(figsize=(12, 8))

    # use different colors
    color_list = ['b', 'c', 'g', 'k', 'm', 'r', 'y']

    for v in range(vehicle_count):
        # get x and y
        x = []
        y = []
        for c in vehicle_tours[v]:
            x.append(x_coordinations[c])
            y.append(y_coordinations[c])

        # plot route
        plt.plot(x, y, c=color_list[v%7], ls="-", lw=0.5, marker='.', ms=10)

    plt.pause(15)
    plt.close()



if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')
