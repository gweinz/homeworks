import sys
import pdb
import math

# reuse code in q1
from q1 import get_baskets_python, get_baskets_spark, triangular_matrix_method, print_frequent_itemsets
from q2 import tuple_list_method
from q2_dict import dict_method
from q3 import get_item_counter

# Assign an index for each item
def get_dict_from_frequent(frequent_list):
    item_dict = {}
    for item in frequent_list:
        item_dict[item] = len(item_dict)
    return item_dict

def aprior_all_method(baskets, support, method, son=False, total_baskets=0):
    # Used by Q5: SON
    if type(baskets) is not list:
        baskets = list(baskets) #baskets are list now
    if son:
        support = math.floor(support*len(baskets)/total_baskets)
        print(f"#total_baskets={total_baskets}, #this_baskets={len(baskets)}, this_support={support}")

    item_counter = get_item_counter(baskets)
    itemsets_1 = sorted([(k, v) for k, v in item_counter.items() if v >= support], key=lambda x: x[1], reverse=True)
    frequent_1 = [x[0] for x in itemsets_1]

    itemsets_list = [itemsets_1]
    frequent_list = frequent_1
    frequent_last = frequent_1

    k = 2
    while True:
        # get a dictionary of current frequent items
        # Note: only frequent item pairs from the last pass is needed
        item_dict = get_dict_from_frequent(frequent_last)

        # baskets will be modfied!
        itemsets = method(baskets, support, item_dict, k=k)
        if len(itemsets) > 0:
            frequent_last = [x[0] for x in itemsets]
            frequent_list += frequent_last
            itemsets_list.append(itemsets)
            k += 1
        else:
            break

    # son method only need items
    if son:
        return frequent_list
    else:
        return itemsets_list

def print_all_frequent_itemsets(itemsets_list):
    n_itemsets = len(itemsets_list)
    for i in range(n_itemsets-1, -1, -1):
        print(f'Frequent itemsets of size: {i+1}')
        print_frequent_itemsets(itemsets_list[i])

# python q4.py 7 lastfm/sample.tsv
if __name__ == '__main__':
    support = int(sys.argv[1])
    filename = sys.argv[2]

    if len(sys.argv) >= 4 and sys.argv[3] == 'spark':
        baskets = get_baskets_spark(filename)
    else:
        baskets = get_baskets_python(filename)

    if len(sys.argv) >= 5:
        if sys.argv[4] == 'matrix':
            method = triangular_matrix_method
        elif sys.argv[4] == 'dict':
            method = dict_method
        else:
            method = tuple_list_method
    else:
        method = tuple_list_method

    # call the algorithm function
    itemsets_list = aprior_all_method(baskets, support, method)
    print_all_frequent_itemsets(itemsets_list)
