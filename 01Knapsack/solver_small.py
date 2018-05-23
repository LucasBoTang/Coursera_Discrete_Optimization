#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # a dynamic programming algorithm for filling the knapsack

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

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
