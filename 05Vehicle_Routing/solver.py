#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import numpy as np
import scipy.spatial.distance as ssd
from matplotlib import pyplot as plt
import math
import random


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    global customer_count, vehicle_capacity
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

    # generate penal
    global penal
    penal = 10 ** (len(str(int(np.max(distances[0])))) + 1) * 2

    # generate initial solution
    initial_generation = initial(demands, distances, vehicle_count)

    # use genetic algorithm
    print('Using genetic algorithm...')
    best_generation = set()
    obj, vehicle_tours, chromesomes = genetic(initial_generation, demands, distances, vehicle_count, True)
    print('best obj: {:2f}'.format(obj))
    best_generation |= chromesomes

    # restart
    for i in range(3):
        print('Restarting...')
        initial_generation = initial(demands, distances, vehicle_count)
        cur_obj, cur_vehicle_tours, chromesomes = genetic(initial_generation, demands, distances, vehicle_count, True)
        best_generation |= chromesomes

        if obj - cur_obj > 0.01:
            obj, vehicle_tours = cur_obj, cur_vehicle_tours
            print('Find a better solution! obj: {:2f}'.format(obj))

    # restart if solution is infeasible
    while obj == float('inf'):
        print('Restarting...')
        initial_generation = initial(demands, distances, vehicle_count)
        obj, vehicle_tours, chromesomes = genetic(initial_generation, demands, distances, vehicle_count, True)

        if obj < float('inf'):
            obj, vehicle_tours = cur_obj, cur_vehicle_tours
            print('Find a better solution! obj: {:2f}'.format(obj))

    # restart with best generation
    print('Restarting with best generation...')
    cur_obj, cur_vehicle_tours, chromesomes = genetic(best_generation, demands, distances, vehicle_count, True)
    if obj - cur_obj > 0.01 :
        obj, vehicle_tours = cur_obj, cur_vehicle_tours
        print('Find a better solution! obj: {:2f}'.format(obj))

    # use variable neighborhood search with best solution
    print('Using variable neighborhood search with best solution...')
    same_best_count = 0
    for i in range(min(100, vehicle_count*10)):
        prev_obj = obj
        obj, vehicle_tours = vns(obj, vehicle_tours, demands, distances, vehicle_count)

        # count when no better solution
        if prev_obj - obj < 0.01:
            same_best_count += 1
        else:
            same_best_count = 0

        # visualize
        if i % 10 == 0：
            visualize(vehicle_tours, x_coordinations, y_coordinations)

        # stop when converge
        if same_best_count > 8 and i > min(vehicle_count*2, 12):
            print('Cannot find a better one!')
            break


    # check feasibility
    if len(vehicle_tours) > vehicle_count:
        print('Solution is infeasible!')
        return

    # visualize solution
    visualize(vehicle_tours, x_coordinations, y_coordinations)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(len(vehicle_tours)):
        outputData += ' '.join([str(customer) for customer in vehicle_tours[v]]) + '\n'

    return outputData



def distanceMatirx(x_coordinations, y_coordinations):

    locations = np.column_stack((x_coordinations, y_coordinations))
    distances = ssd.cdist(locations, locations, 'euclidean')

    return np.array(distances)



def initial(demands, distances, vehicle_count):

    initial_generation = set()

    chromesome = greedy(demands, distances, 0, vehicle_count)
    initial_generation.add(chromesome)

    for _ in range(4):
        chromesome = greedy(demands, distances, 0.2, vehicle_count)
        initial_generation.add(chromesome)

    return initial_generation


def greedy(demands, distances, probability, vehicle_count):

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
            # reject with probability
            random_num = random.random()
            if random_num < probability:
                i += 1
                continue
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
            chromesome.append(i)
            cur_customer = i
            cur_capacity -= demands[i]
            used.add(i)

        # when one tour is completed
        cur_customer = 0
        cur_capacity = vehicle_capacity

    chromesome = tuple(chromesome)

    return chromesome



def genetic(initial_generation, demands, distances, vehicle_count, claim):

    # initialize parameters
    chromesome_len = len(list(initial_generation)[0])
    G = 150 # steps of generations
    P = min(120, math.factorial(chromesome_len)) # size of population
    S = min(100, math.factorial(chromesome_len)) # number of survivals
    pc = 0.95 # probability of crossover
    pm = 0.2 # probability of mutation

    # initialize first generation
    generation = generationInitialze(initial_generation, P)

    g = 0 # count generations
    same_best_count = 0 # count when the best solution is same as previous
    prev_best_obj = float('inf')

    best_objs = []
    medium_objs = []
    worst_objs = []

    # stop when solution is converge and steps of generations achive the goal
    while g < G or same_best_count < 80:

        # decode to get phenotype
        phenotype_genotype = getPhenotype(generation, demands, distances, vehicle_count)
        best_objs.append(phenotype_genotype[0][0])
        medium_objs.append(phenotype_genotype[P//2][0])
        worst_objs.append(phenotype_genotype[-1][0])

        # if current best solution is same as previous best solution
        if phenotype_genotype[0][0] == prev_best_obj:
            same_best_count += 1
        else:
            same_best_count = 0
        prev_best_obj = phenotype_genotype[0][0]

        # eliminate
        phenotype_genotype = phenotype_genotype[:S]

        # generate offsprings
        generation = offspringGenerate(phenotype_genotype, P, pc, pm, vehicle_count)

        # apply elitism
        for i in range(5):
            generation.add(phenotype_genotype[i][2])

        # count generations
        g += 1
        if claim and g % 200 == 0:
            print('{} generations...'.format(g))

    # decode the last generation
    phenotype_genotype = getPhenotype(generation, demands, distances, vehicle_count)
    best_objs.append(phenotype_genotype[0][0])
    medium_objs.append(phenotype_genotype[P//2][0])
    worst_objs.append(phenotype_genotype[-1][0])
    g += 1

    # visualize obj for each generations
    if claim:
        plt.ion()
        plt.xlabel("Generation")
        plt.ylabel("Cost")
        plt.plot(range(g), best_objs, c='g', lw=1)
        plt.plot(range(g), medium_objs, c='y', lw=1)
        plt.plot(range(g), worst_objs, c='r', lw=1)
        plt.pause(5)
        plt.close()

    # avoid infeasible
    i = 0
    while len(phenotype_genotype[i][1]) > vehicle_count:
        i += 1
        if i == S:
            print('All individuals are infeasible!')
            return float('inf'), phenotype_genotype[0][1], set()

    return phenotype_genotype[i][0], phenotype_genotype[i][1], set([individual[2] for individual in phenotype_genotype[:S//5]])



def generationInitialze(initial_generation, P):

    lst = list(list(initial_generation)[0])

    while len(initial_generation) < P:
        random.shuffle(lst)
        chromesome = tuple(lst)
        initial_generation.add(chromesome)

    return initial_generation



def getPhenotype(generation, demands, distances, vehicle_count):

    # initial array
    phenotype_genotype = []

    # decode chromesome
    for chromesome in generation:
        obj, vehicle_tours = decode(chromesome, demands, distances, vehicle_count)
        phenotype_genotype.append([obj, vehicle_tours, chromesome])

    # sort obj
    phenotype_genotype.sort(key=lambda x: x[0])

    return phenotype_genotype



def decode(chromesome, demands, distances, vehicle_count):

    # initial vehicle tour
    vehicle_tours = []


    # start with depot
    obj = 0
    cur_customer = 0
    vehicle_index = 1
    cur_capacity = vehicle_capacity
    vehicle_tour = [0]

    for c in chromesome:
        # cheack capacity
        if demands[c] > cur_capacity:
            # go back to depot
            obj += distances[cur_customer, 0]
            cur_capacity = vehicle_capacity
            vehicle_tour.append(0)
            vehicle_tours.append(vehicle_tour)
            # assign a new vehicle
            vehicle_index += 1
            vehicle_tour = [0]
            obj += distances[0, c]
        else:
            obj += distances[cur_customer, c]
        # update
        cur_capacity -= demands[c]
        cur_customer = c
        vehicle_tour.append(c)

    # go back to depot finally
    obj += distances[cur_customer, 0]
    vehicle_tour.append(0)
    vehicle_tours.append(vehicle_tour)

    # penalize if infeasible
    if vehicle_index > vehicle_count:
        obj += penal * (vehicle_index - vehicle_count)
    else:
        for _ in range(vehicle_count - vehicle_index):
            vehicle_tours.append([0, 0])

    return obj, vehicle_tours



def offspringGenerate(phenotype_genotype, P, pc, pm, vehicle_count):

    # calculate fitness by inversing obj
    fitnesses = 1 / np.array([individual[0] for individual in phenotype_genotype])

    # generate offsprings
    offsprings = set()
    while len(offsprings) < P:

        # get parents by roulette wheel selection
        parent1 = phenotype_genotype[roulettewheel(fitnesses)][2]
        parent2 = phenotype_genotype[roulettewheel(fitnesses)][2]

        # cross over
        random_num = random.random()
        if random_num < pc:
            offspring1, offspring2 = crossover(parent1, parent2)
        else:
            offspring1, offspring2 = parent1, parent2

        # mutate
        random_num = random.random()
        if random_num < pm:
            offspring1 = mutate(offspring1)
        random_num = random.random()
        if random_num < pm:
            offspring2 = mutate(offspring2)

        offsprings.add(offspring1)
        offsprings.add(offspring2)

    return offsprings



def roulettewheel(fitnesses):

    total = np.sum(fitnesses)
    random_num = random.uniform(0, total)
    cur_index = 0
    cur_sum = 0

    for fitness in fitnesses:
        cur_sum += fitness
        if random_num < cur_sum:
            return cur_index
        else:
            cur_index += 1



def crossover(parent1, parent2):

    # get cut index
    cut1, cut2 = sorted(random.sample(range(customer_count - 1) , 2))
    cut2 += 1

    # cut segment
    seg1 = parent1[cut1:cut2]
    seg2 = parent2[cut1:cut2]

    # swap segment
    offspring1 = segSwap(parent1, seg2, cut1, cut2)
    offspring2 = segSwap(parent2, seg1, cut1, cut2)

    return offspring1, offspring2



def segSwap(parent, seg, cut1, cut2):

    offspring = tuple()

    i, j = 0, 0
    for c in parent:
        j += 1
        if c not in seg:
            offspring += (c,)
            i += 1
        if i == cut1:
            break

    offspring += seg

    for c in parent[j:]:
        if c not in seg:
            offspring += (c,)

    return offspring



def mutate(offspring):

    # get cut index
    cut1, cut2 = sorted(random.sample(range(customer_count - 1) , 2))
    cut2 += 1

    # applt 2-opt
    return offspring[:cut1] + offspring[cut1:cut2][::-1] + offspring[cut2:]



def vns(obj, vehicle_tours, demands, distances, k_max):

    # initialize
    k = 1
    routes = []
    chromesome = tuple()
    part_obj = 0

    # choose a route randomly
    route = random.randint(0, k_max-1)
    # avoid first route is unused or have less than two customers
    while len(vehicle_tours[route]) <= 4:
        route = random.randint(0, k_max-1)

    while k < k_max:

        # avoid choose same route
        while route in routes:
            route = random.randint(0, k_max-1)

        routes.append(route)
        # get chromesome and obj of chosen part
        chromesome += tuple(vehicle_tours[route][1:-1])
        part_obj += decode(tuple(vehicle_tours[route]), demands, distances, 1)[0]

        # generate initial generation
        initial_generation = set()
        initial_generation.add(chromesome)

        # search by genetic algorithm
        cur_part_obj, cur_vehicle_tours, _ = genetic(initial_generation, demands, distances, k, False)

        # stop when a better solution
        if part_obj - cur_part_obj > 0.01:
            # update obj
            obj = obj - part_obj + cur_part_obj
            # update current routes
            for i in range(k):
                vehicle_tours[routes[i]] = cur_vehicle_tours[i]
            print('Find a better solution! obj: {:2f}'.format(obj))

            return obj, vehicle_tours

        k += 1

    # when no better solution
    return obj, vehicle_tours



def visualize(vehicle_tours, x_coordinations, y_coordinations):

    plt.ion()
    plt.figure(figsize=(12, 8))

    # use different colors
    color_list = ['b', 'c', 'g', 'm', 'r', 'y']

    for v in range(len(vehicle_tours)):
        # get x and y
        x = []
        y = []
        for c in vehicle_tours[v]:
            x.append(x_coordinations[c])
            y.append(y_coordinations[c])

        # plot route
        plt.plot(x, y, c=color_list[v%6], ls="-", lw=0.5, marker='.', ms=8, alpha = 0.8)

    # plot depot
    plt.scatter(x_coordinations[0], y_coordinations[0], c='k', marker='p', s=100)

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
