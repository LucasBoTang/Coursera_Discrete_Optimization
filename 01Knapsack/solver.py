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

    # show the capaicty and number of items
    print('Capacity:', str(capacity))
    print('Number of Items:', str(item_count))

    if item_count > 5000:
        # a trivial greedy algorithm for filling the knapsack
        # it takes items in-order until the knapsack is full
        print('Using greedy algorithm...')
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

    # a dynamic programming algorithm for filling the knapsack
    # use PyTables to store large matrix
    print('Using dynamic programming...')
    import tables as tb
    import numpy as np
    store = 'store.h5'
    filters = tb.Filters(complevel=5, complib='blosc') # use BLOSC compression
    hdf5_file = tb.open_file(store, mode='w')
    dp_table = hdf5_file.create_carray(hdf5_file.root, 'data',
                                      tb.Int32Atom(),
                                      shape=(capacity+1, item_count+1),
                                      filters=filters)

    # initialize the first column
    prev = dp_table[:, 0] = np.zeros(capacity+1)
    cur = np.empty(capacity+1)

    # fill the table
    for i in range(1, item_count+1):
        # when the weight of the item is greater than the capacity
        if items[i-1].weight > capacity:
            cur = prev
        # when the weight of the item is less than or equal to the capacity
        else:
            # the remain capacity cannot satisfy the item
            cur[:items[i-1].weight] = prev[:items[i-1].weight]
            # the remain capacity can satisfy the item
            cur[items[i-1].weight:] = np.maximum(prev[items[i-1].weight:], \
                                                 items[i-1].value+prev[:-items[i-1].weight])
        dp_table[:, i] = cur
        prev = cur

        # show the remaining items
        if (item_count - i) % 100 == 0:
            print('{} items remain...'.format(item_count-i))

    # close the file
    hdf5_file.close()

    # read the file
    read_hdf5_file = tb.open_file(store, mode='r')
    dp_table = read_hdf5_file.root.data

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

    # close the file
    read_hdf5_file.close()

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
