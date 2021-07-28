import pandas as pd
import networkx as nx
import json

#read in the json file
tweets = []
with open("Gamergate.json", "r", encoding="utf-8") as f:
    for line in f.readlines():
        J = json.loads(line)

        tweets.append(J)

print(tweets[0])
#create the multigraph
# G = nx.MultiGraph()        
# for movie in Movies:
#     for i in range(0, len(movie["cast"]) - 1):
#         for j in range(i + 1, len(movie["cast"])):
#             G.add_edge(movie["cast"][i], movie["cast"][j], title = movie["title"])
# print("Number of nodes in this multigraph =", G.number_of_nodes())
# print("Number of edges in this multigraph =", G.number_of_edges())

# #save the graph as a gxef
# nx.write_gexf(G, "movie_graph.gexf")
