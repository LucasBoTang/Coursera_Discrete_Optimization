#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
import cvxopt as op


Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    global facility_count, customer_count
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    facilities_location = np.zeros((facility_count, 2))
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))
        facilities_location[i-1] = [float(parts[2]), float(parts[3])]

    customers = []
    customers_location = np.zeros((customer_count, 2))
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))
        customers_location[i-facility_count-1] = [float(parts[1]), float(parts[2])]

    # show the size of problem
    print('\n')
    print('Number of facilities:', facility_count)
    print('Number of customers:', customer_count)

    # build distance matirx
    print('Building distance matirx...')
    distances = distanceMatirx(facilities_location, customers_location)

    if customer_count <= 100:
        # construct matrix for linear relaxation
        print('Coverting data into tableau...')
        A_ub, b_ub, A_eq, b_eq, c = tableauConstruction(facilities, customers, distances)
        # use branch and bound algorithm
        print('Using branch and bound algorithm with linear relaxation...')
        global obj, solution
        obj = float('inf')
        solution = [-1] * customer_count
        branch_bound(facilities, customers, distances, A_ub, b_ub, A_eq, b_eq, c)

    else:
        # start with greedy algorithm
        print('Starting with greedy algorithm...')
        obj, solution = greedy(facilities, customers, distances)

    # visualize
    visualize(facilities, customers, solution)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def distanceMatirx(facilities_location, customers_location):

    # initialize the distance matrix
    distances = np.zeros((facility_count, customer_count))

    # calculate the distance between customers and facilities
    for i in range(customer_count):
        cur_customer = np.array([customers_location[i]])
        distances[:, i] = np.sqrt(np.square(cur_customer[:, 0] - facilities_location[:, 0]) + \
                                  np.square(cur_customer[:, 1] - facilities_location[:, 1]))

    return distances


def branch_bound(facilities, customers, distances, A_ub, b_ub, A_eq, b_eq, c):

    # calculate linear relaxation
    global obj, solution
    lp_relexation = op.solvers.lp(c, A_ub, b_ub, A_eq, b_eq)

    # bound
    if lp_relexation['primal objective'] > obj:
        return

    # find the most fractional variable
    x = np.array(lp_relexation['x']).reshape(-1)
    fra_var = np.argsort(np.abs(x - 0.5))[0]

    # find an integer solution
    if x[fra_var] < 0.0001 or abs(x[fra_var] - 1) < 0.0001:
        cur_solution = [-1] * customer_count
        for i in range(facility_count):
            for j in range(customer_count):
                if abs(x[facility_count+customer_count*i+j] - 1) < 0.0001:
                    cur_solution[j] = i
        cur_obj = objCal(facilities, customers, distances, cur_solution)
        if cur_obj < obj:
            obj, solution = cur_obj, cur_solution
        return

    # branch
    new_A_eq = np.zeros((1, facility_count+facility_count*customer_count))
    new_A_eq[0, fra_var] = 1
    new_A_eq = np.concatenate((np.array(A_eq), new_A_eq), axis=0)
    new_A_eq = op.matrix(new_A_eq)
    # do not take the variable
    new_b_eq = np.concatenate((np.array(b_eq).T[0], [0]))
    new_b_eq = op.matrix(new_b_eq)
    branch_bound(facilities, customers, distances, A_ub, b_ub, new_A_eq, new_b_eq, c)
    # take the variable
    new_b_eq = np.concatenate((np.array(b_eq).T[0], [1]))
    new_b_eq = op.matrix(new_b_eq)
    branch_bound(facilities, customers, distances, A_ub, b_ub, new_A_eq, new_b_eq, c)


def greedy(facilities, customers, distances):

    # initialize
    solution = [-1] * customer_count
    capacity_remaining = [facility.capacity for facility in facilities]

    # avoid expensive facilities
    ban_list = np.argsort([facility.setup_cost for facility in facilities])[::-1][:facility_count//10]

    # assign customer to nearest avaliable facility
    for c in range(customer_count):
        sorted_facilities = np.argsort(distances[:, c])

        i = 0
        f = sorted_facilities[i]
        while f in ban_list or customers[c].demand > capacity_remaining[f]:
            i += 1
            f = sorted_facilities[i]

        capacity_remaining[f] -= customers[c].demand
        solution[c] = f

    obj = objCal(facilities, customers, distances, solution)

    return obj, solution


def objCal(facilities, customers, distances, solution):

    # generate a statistics of the use of facilities
    used = [0] * facility_count
    for facility_index in solution:
        used[facility_index] = 1

    # calculate the cost of the solution
    obj = sum([facility.setup_cost * used[facility.index] for facility in facilities])
    for c in range(customer_count):
        f = solution[c]
        obj += distances[f, c]

    return obj


def tableauConstruction(facilities, customers, distances):

    # initialize A_ub, b_ub, A_eq, b_eq and c
    A_ub = np.zeros((facility_count*customer_count+facility_count, facility_count+facility_count*customer_count))
    b_ub = np.zeros(facility_count*customer_count+facility_count)
    A_eq = np.zeros((customer_count, facility_count+facility_count*customer_count))
    b_eq = np.zeros(customer_count)
    c = np.zeros(facility_count+facility_count*customer_count)

    # set c
    c[:facility_count] = [facility.setup_cost for facility in facilities]
    c[facility_count:] = distances.reshape(-1)
    c = op.matrix(c)

    # set A_ub
    demand = [customer.demand for customer in customers]
    for i in range(facility_count):
        A_ub[customer_count*i:customer_count*(i+1), i] = [-1] * customer_count
        A_ub[facility_count*customer_count+i, facility_count+customer_count*i:facility_count+customer_count*(i+1)] = demand
    A_ub[:facility_count*customer_count, facility_count:] = np.eye(facility_count*customer_count)
    A_ub = np.concatenate((A_ub, np.eye(facility_count+facility_count*customer_count), -np.eye(facility_count+facility_count*customer_count)), axis=0)
    A_ub = op.matrix(A_ub)

    # set b_ub
    capacity = [facility.capacity for facility in facilities]
    b_ub[facility_count*customer_count:] = capacity
    b_ub = np.concatenate((b_ub, np.ones(facility_count+facility_count*customer_count), np.zeros(facility_count+facility_count*customer_count)))
    b_ub = op.matrix(b_ub)

    # set A_eq
    for i in range(customer_count):
        temp = np.zeros((facility_count, customer_count))
        temp[:, i] += 1
        temp = temp.reshape(-1)
        A_eq[i, facility_count:] = temp
    A_eq = op.matrix(A_eq)

    # set b_eq
    b_eq += 1
    b_eq = op.matrix(b_eq)

    return A_ub, b_ub, A_eq, b_eq, c


def visualize(facilities, customers, solution):

    plt.figure(figsize=(12, 8))

    # use different colors
    color_list = ['b', 'c', 'g', 'k', 'm', 'r', 'y']

    # connect customer into facility
    for i in range(customer_count):

        # get customers and facilities
        customer = customers[i]
        facility = facilities[solution[i]]

        # get x and y
        x = [customer.location[0], facility.location[0]]
        y = [customer.location[1], facility.location[1]]

        # plot
        plt.plot(x, y, c=color_list[solution[i]%7], ls="-", lw=0.2, marker='.', ms=2)

    # emphasize facilities
    for i in range(facility_count):

        # get x and y
        facility = facilities[i]
        x, y = facility.location[0], facility.location[1]

        # plot
        plt.scatter(x, y, c=color_list[i%7], marker='p', s=20)

    plt.show()


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')
