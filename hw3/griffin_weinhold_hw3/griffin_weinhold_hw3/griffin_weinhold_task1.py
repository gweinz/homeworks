import pandas as pd
import networkx as nx
import json
import matplotlib.pyplot as plt
from operator import itemgetter
import sys


if __name__ == "__main__":
    input_file = sys.argv[1]
    gexf_output = sys.argv[2]
    json_output = sys.argv[3]
    tweets = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            tweet = json.loads(line)

            tweets.append(tweet)

    weight_map = {}
    G = nx.DiGraph()
    for t in tweets:
        if 'text' in t:
            usr = t['user']['screen_name']
            G.add_node(usr)

            if 'retweeted_status' in t:

                rt = t['retweeted_status']['user']['screen_name']

                if (usr, rt) in weight_map:
                    weight_map[(usr, rt)] += 1

                else:
                    weight_map[(usr, rt)] = 1

                G.add_edge(usr, rt, weight = weight_map[(usr, rt)])

    nx.write_gexf(G, gexf_output)


    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()


    max_edge = -1
    max_node = ''
    end = ''

    outgoing = {}
    incoming = {}

    for u, v, weight in G.edges(data="weight"):
        if weight is not None:
            
            if u in outgoing:
                outgoing[u] += weight
            else:
                outgoing[u] = weight
                
            if v in incoming:
                incoming[v] += weight
            else:
                incoming[v] = weight
            
            if weight > max_edge:
                max_edge = weight
                max_node = u
                end = v
                
    max_rtd = max(incoming, key=incoming.get)
    max_rtr = max(outgoing, key=outgoing.get)
    max_rtd_num = G.in_degree(max(incoming, key=incoming.get), weight='weight')
    max_rtr_num = G.out_degree(max(outgoing, key=outgoing.get), weight='weight')

    result = {}

    result['n_nodes'] = num_nodes
    result['n_edges'] = num_edges

    result['max_retweeted_user'] = max_rtd
    result['max_retweeted_number'] = max_rtd_num

    result['max_retweeter_user'] = max_rtr
    result['max_retweeter_number'] = max_rtr_num


    with open(json_output, 'w') as outfile:
        json.dump(result, outfile)


