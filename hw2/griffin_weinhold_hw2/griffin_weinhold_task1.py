import sys
import pdb
import collections
import pandas as pd 
import itertools
import bisect
import json
from pyspark import SparkContext



def tuple_wrapper(s):
    if type(s) is not tuple:
        s = (s, )
    return s
def get_baskets_spark_rdd(filename, sc, partition):
    #.flatMap expect a list of items is returned. Could be zero or many items. Each item would be in format: (user_id, artist_id)
    #.groupByKey assume the first element is the key. Items having the same key will be merged to a list, the result format is: (user_id, [artist_id1, artist_id2, artist_id3, ...])
    baskets_rdd = sc.textFile(filename, partition)\
        .flatMap(process_line)\
        .groupByKey()

    return baskets_rdd

def get_dict_from_frequent(frequent_list):
    item_dict = {}
    for item in frequent_list:
        item_dict[item] = len(item_dict)
    return item_dict


def get_baskets_spark(filename):
    # Can't use the same logic as in pure python, because spark will read file in parallel.
    # Use groupByKey to merge artist_id liked by the same user_id into a list.
    # Another benfits is, this logic allows the file lines been shuffled, which is not allowed in the python logic
    sc = SparkContext("local","PySpark Tutorial") #In Q1, Q2, Q3, Q4 spark is only used for reading data

    # Use only 1 core
    baskets = get_baskets_spark_rdd(filename, sc, 1).collect()

    sc.stop()
    return baskets


class FirstList(collections.UserList):
    def __lt__(self, other):
        return self[0].__lt__(other)
def inverse_dict(d):
    # {key: value} will become {value: key}
    return {v: k for k, v in d.items()}
def get_possible_k(item_dict, k):
    possible_k = {}
    for pair in itertools.combinations(item_dict.keys(), 2):
        pair_set = set()
        for i in range(2):
            pair_set = pair_set.union(tuple_wrapper(pair[i]))
        if len(pair_set) == k:
            possible_k[frozenset(pair_set)] = [pair[0], pair[1]]
    return possible_k

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

def filter_basket(baskets, item_dict, k):
    if k == 2:
        possible_item = item_dict
    else:
        possible_item = set()
        possible_item = possible_item.union(*item_dict.keys())

    for i in range(len(baskets)):
        basket = baskets[i]
        items = basket[1]
        items_filterd = [item for item in items if item in possible_item]
        baskets[i] = (basket[0], items_filterd)

def tuple_list_method(baskets, support, item_dict=None, k=5):
    if item_dict is None:
        item_dict = get_item_dict(baskets)
    else:
     
        filter_basket(baskets, item_dict, k)

    item_dict_inv = inverse_dict(item_dict)
    n = len(item_dict)

    # Only used in Q3, Q4, Q5
    if k >= 3:
        possible_k = get_possible_k(item_dict, k)

    tuples = [] # Storage space is allocated every time a new pair is occurred, similar to LinkedList

    # Key logic: Tuple List Method
    for basket in baskets:
        items = basket[1]

        for kpair in itertools.combinations(items, k):
            # kpair is a k element tuple, kpair[i] is item (string)
            if k >= 3:
                pair_set = frozenset(kpair)

                # Now kpair is a 2 element pair
                kpair = possible_k.get(pair_set, None)
                if kpair is None:
                    continue

            i = item_dict[kpair[0]]
            j = item_dict[kpair[1]]

            if i > j:
                j, i = i, j
         
            idx = i*n+j

           
            insert_idx = bisect.bisect_left(tuples, idx)

            # The insertion index is at the end of the list, i.e. the new item is larger than all items in the list
            if insert_idx >= len(tuples):
                tuples.append(FirstList([idx, 1]))
            else:
                tp = tuples[insert_idx]

                # This pair is already in the tuple list. Increase it's count (second element) by 1
                if tp[0] == idx:
                    tp[1] += 1
                else:
                    # This pair is not yet in the tuple list. Add a new tuple, the format is: (1D index, count)
                    tuples.insert(insert_idx, FirstList([idx, 1]))

    # Extract results
    frequent_itemset_list = []
    for tp in tuples:
        count = tp[1]

        # Convert 1D index to 2D index
        # If you use different indexing method, this also needs to be changed
        i = tp[0] // n
        j = tp[0] % n

        item_i = item_dict_inv[i]
        item_j = item_dict_inv[j]

        # This implementation is ready for k>=3
        item_all = set()
        for item in (item_i, item_j):
            item_all = item_all.union(tuple_wrapper(item))

        item_all = tuple(sorted(list(item_all)))

        # apply support threshold
        if count >= support:
            frequent_itemset_list.append((item_all, count))

    frequent_itemset_list = sorted(frequent_itemset_list, key=lambda x: [-x[1]] + list(x[0]))
    return frequent_itemset_list



def process_line(line):
   
    tokens = [t for t in line.split(',') if t != '']


    if len(tokens) == 4:
        if tokens[2] == '5.0':

            return [tokens[:2]] #one-element list.

        else:
            return []
    else:
        return []

# Count occurrence of each item
def get_item_counter(baskets):
    item_counter = collections.Counter()
    for basket in baskets:
        items = basket[1]
        item_counter.update(items)
    return item_counter

# same as get_item_dict in q1.py, except there is a threshold of support
def get_item_dict_threshold(item_counter, support):
    item_dict = {}
    for k, v in item_counter.items():
        if v >= support:
            item_dict[k] = len(item_dict)
    return item_dict

def aprior_method(baskets, support, method):
    item_counter = get_item_counter(baskets)

    item_dict = get_item_dict_threshold(item_counter, support)
   


    return item_counter, method(baskets, support, item_dict)

def get_support_dict(itemsets):

    i_dict = {}

    for items in itemsets:
        for item in items:
            i_dict[item[0]] = item[1]

    return i_dict

def get_assoc(itemsets_list, item_counter, supports, interest_threshold):
    results = []
    idx = 1
    for items in itemsets_list:
  
        if idx > 1:
            for item in items:
                idx+=1
     
                recs = list(item[0])
                union_support = item[1]
                
                single = False
            
                for j in recs:

                    I = list(recs)
                    

                    I.remove(j)
                    
                    if len(recs) == 2:
                        I_support = item_counter[I[0]]
                   
                    else:
                        I_support=supports[tuple(I)]
                        

             
                    conf = (union_support/I_support)
                    interest = conf - (item_counter[j]/num)
                    if interest > interest_threshold:
                        newI = [int(i) for i in I]
                        newI.sort()
                        results.append((newI, int(j), interest, union_support))

        else:
            idx+=1
    
    sorted_results = sorted(results, key=lambda x: (-abs(x[2]), -x[3], x[0], x[1]))
    return sorted_results

# python q3.py 10 lastfm/sample.tsv
if __name__ == '__main__':
    
    filename = sys.argv[1]
    writefile = sys.argv[2]
    interest_threshold = float(sys.argv[3])
    support= int(sys.argv[4])

    baskets = get_baskets_spark(filename)
    # baskets = get_baskets_python(filename)
 

    method = tuple_list_method

   
    
    df = pd.read_csv(filename)

    num = len(baskets)

    item_counter = get_item_counter(baskets)


  
    itemsets_list = aprior_all_method(baskets, support, method)
    supports = get_support_dict(itemsets_list)

    
    results = get_assoc(itemsets_list, item_counter, supports, interest_threshold)
    

    with open(writefile, 'w', encoding='utf8') as outfile:
   

        json.dump(results, outfile)
       
