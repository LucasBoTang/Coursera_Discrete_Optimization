#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    global item_count, capacity
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    global items
    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # show the capaicty and the number of items
    print('\n')
    print('Capacity:', str(capacity))
    print('Number of Items:', str(item_count))

    # a dynamic programming algorithm for filling the knapsack
    if item_count * capacity <= 100000000:
        print('Using dynamic programming...')

        # initialize a table
        import numpy as np
        dp_table = np.zeros((capacity+1, item_count+1))

        # fill the table
        for i in range(1, item_count+1):
            # when the weight of the item is greater than the capicity
            if items[i-1].weight > capacity:
                dp_table[:, i] = dp_table[:, i-1]
            # when the weight of the item is less than or equal to the capicity
            else:
                # the remain capicity cannot satisfy the item
                dp_table[:items[i-1].weight, i] = dp_table[:items[i-1].weight, i-1]
                # the remain capicity can satisfy the item
                dp_table[items[i-1].weight:, i] = np.maximum(dp_table[items[i-1].weight:, i-1], \
                                                             items[i-1].value+dp_table[:-items[i-1].weight, i-1])

        # get the opitimal value
        value = int(dp_table[-1, -1])

        # initialize the taken list
        taken = [0] * item_count

        # trace back
        remain_weight = capacity
        for i in range(1, item_count+1):
            # when the weight between previous and current is different, the item should be taken
            if dp_table[remain_weight, -i] != dp_table[remain_weight, -i-1]:
                taken[-i] = 1
                remain_weight -= items[-i].weight

        # prepare the solution in the specified output format
        output_data = str(value) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, taken))
        return output_data

    # a branch&bound algorithm for filling the knapsack
    print('Using branch and bound...')

    # sort items by value density for relaxation
    items.sort(key=lambda item: item.value / item.weight, reverse=True)

    # initialize the branches count
    global branch_count
    branch_count = 0

    # initialize the value and taken
    global max_value, max_items
    max_value = 0
    max_items = []
    value_bound = relaxation(0, 0, capacity)
#    value_bound = 0 # no relaxation
#    for item in items:
#        value_bound += item.value

    # depth-first search
    try:
        import sys
        sys.setrecursionlimit(3000)
        dfs(0, 0, [], capacity, value_bound)

        # get the optimal value and taken
        value = max_value
        taken = [0] * item_count
        for item in max_items:
            taken[item] = 1

    # a trivial greedy algorithm for filling the knapsack when stack overflow
    except:
        print('Stack overflow! Using greedy algorithm...')
        value = 0
        weight = 0
        taken = [0] * item_count

        for item in items:
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

# define a depth-first search for branch&bound
def dfs(i, cur_value, cur_items, remain_weight, value_bound):

    global branch_count
    global max_value, max_items

    # check the feasibility
    if remain_weight < 0:
#        print('---', remain_weight, '---')
        branch_count += 1
        if (branch_count % 100000000) == 0:
            print('{} branches has been calculated...'.format(branch_count))
        return

#    print(cur_value, remain_weight, value_bound)

    # prune branch with bound
    if value_bound < max_value:
        branch_count += 1
        if (branch_count % 100000000) == 0:
            print('{} branches has been calculated...'.format(branch_count))
        return

    # update the current feasible max value
    if cur_value >= max_value:
        max_value, max_items = cur_value, cur_items

    # stop when all of items are used
    if i == item_count:
        branch_count += 1
        if (branch_count % 100000000) == 0:
            print('{} branches has been calculated...'.format(branch_count))
        return

    # take the item
    dfs(i+1, cur_value+items[i].value, cur_items[:]+[items[i].index], remain_weight-items[i].weight, value_bound)

    # do not take the item
    relaxation_bound = relaxation(i+1, cur_value, remain_weight)
    dfs(i+1, cur_value, cur_items, remain_weight, relaxation_bound)
#    dfs(i+1, cur_value, cur_items, remain_weight, value_bound-items[i].value) # no relaxation

def relaxation(i, cur_value, remain_weight):

    # initialize
    cur_weight = 0
    relaxation_bound = cur_value
    weight_add = value_add = 0

#    print('Basic:', i, cur_value)

    # fill the capacity with heighest value density
    while cur_weight + weight_add <= remain_weight:

        cur_weight += weight_add
        relaxation_bound += value_add

        # when all of items are used
        if i == item_count:
            return relaxation_bound

        else:
            weight_add = items[i].weight
            value_add = items[i].value
            i += 1
#            print('Add:', i, value_add)


    # fill by fraction
    relaxation_bound += (remain_weight - cur_weight) * value_add / weight_add

    return relaxation_bound

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
