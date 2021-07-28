import pandas as pd
import networkx as nx
import json, sys
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import time


def remove(G):
    
    all_edges = nx.edge_betweenness_centrality(G, weight='weight')
   
    maxx = max(all_edges.values())      
    
    edges=[]
    for key, value in sorted(all_edges.items(), key=lambda item: item[1], reverse = True):
        if value == maxx:
            edges.append(key)
            
        else:
            break
        
            
    return edges

def girvan_newman(G, OG):
    

    splits = nx.number_connected_components(G)
    edges = G.number_of_edges()
    max_mod = -100
    
    while(edges > 1):
        
        
        # print('edges left', edges)
        to_remove = remove(G)
        
        for t in to_remove:
            G.remove_edge(t[0], t[1])
        
        edges = G.number_of_edges()
        
            
       
        old_splits = splits
        splits = nx.number_connected_components(G)
        
        
        if old_splits != splits:
            # print('components', splits)
            
            modularity = get_modularity(G, OG)
           
        
            if modularity > max_mod:
                
                
                max_mod = modularity
                # print('new max mod:', max_mod)
                
                id_splits = splits
                partitions = list(nx.connected_components(G))
        
    return partitions, max_mod, id_splits

def get_modularity(G, OG):
    M = OG.size('weight')
    cumulative_modularity = 0

    partitions = nx.connected_components(G)
    
    for parts in partitions:
        part_modularity = 0

        for i in parts:
            for j in parts:
                
                if i != j:
                    
                    ki = OG.degree(i, weight='weight')
                    kj = OG.degree(j, weight='weight')
                    
                    A_bool = G.has_edge(i,j)

                    if A_bool:
                        Aij = G[i][j]['weight']
                    else:
                        Aij = 0
                    
                    rh = (ki*kj)/(2*M)
                    tmp_modularity = Aij - rh
                    
                    
                    part_modularity += tmp_modularity
                    
        
        cumulative_modularity += part_modularity
          
    return cumulative_modularity/(2*M)    

def taskA(tweets, output):
            

    weight_map = {}
    tweet_map = {}
    G = nx.Graph()
    OG = nx.Graph()
    for t in tweets:
       
        if 'text' in t:
            usr = t['user']['screen_name']
            text = t['text']
            
            G.add_node(usr)
            OG.add_node(usr)
            if 'retweeted_status' in t:

                rt = t['retweeted_status']['user']['screen_name']
               
               
                if (usr, rt) in weight_map:
                    weight_map[(usr, rt)] += 1

                else:
                    weight_map[(usr, rt)] = 1

                G.add_edge(usr, rt, weight = weight_map[(usr, rt)])
                OG.add_edge(usr, rt, weight = weight_map[(usr, rt)])


    split_list, modularity_score, c_num = girvan_newman(G, OG)


    split_list.sort(key=len)
    results = []
    for s in split_list:
        results.append(sorted(list(s)))

    rst = sorted(results, key=itemgetter(0))
    rst.sort(key=len)

    with open(output, 'w') as filehandle:
        filehandle.write("Best Modularity is: " + str(modularity_score) + '\n')
        filehandle.writelines("%s\n" % str(r)[1:-1].replace(" ", "") for r in rst)

    return rst

def taskB(tweets, components, output):
    tweet_map = {}

    for t in tweets:

        if 'text' in t:
            usr = t['user']['screen_name']
            text = t['text']
            if usr in tweet_map:
                tweet_map[usr] += ' ' + text
            else:
                tweet_map[usr] = text

            if 'retweeted_status' in t:

                rt = t['retweeted_status']['user']['screen_name']
                    
                rtext = t['retweeted_status']['text']
                if rt in tweet_map:
                    tweet_map[rt] += ' ' + rtext

                else:
                    tweet_map[rt] = rtext
                   
    
    k_comm = components[-2:]
    rest = components[:-2]
    arr = []

    cluster = 1
    for k in k_comm:
        for usr in k:
            tweet = tweet_map[usr]
                
            bit = (usr, tweet, cluster)
            arr.append(bit)
        cluster -=1

    df = pd.DataFrame(arr, columns=['user', 'tweet', 'community'])
    train_len = len(df)

    arr = []

    for k in rest:
        for usr in k:
            tweet = tweet_map[usr]

            bit = (usr, tweet, -1)
            arr.append(bit)
    df2 = pd.DataFrame(arr, columns=['user', 'tweet', 'community'])
    df = df.append(df2)


    train = df[:train_len]
    test = df[train_len:]

    vectorizer = TfidfVectorizer()

    features_train = train['tweet']
    labels_train = train['community']
    features_test = test['tweet']
    # labels_test = test['community']

    features_train = vectorizer.fit_transform(features_train)

    clf = MultinomialNB()
    clf.fit(features_train, labels_train)
    
    features_test = vectorizer.transform(features_test)
    preds = clf.predict(features_test)
    test['community'] = preds
    train['community'] = clf.predict(features_train)
    final = test.append(train)


    results_map = {}
    results_map[0] = []
    results_map[1] = []

    for index, row in final.iterrows():
        results_map[row['community']].append(row['user'])

    with open(output, "w", encoding="utf-8") as f:
        for k in sorted(results_map, key=lambda k: len(results_map[k]), reverse=False):
            result = results_map[k]
            push = sorted(result)
            f.write("%s\n" % str(push)[1:-1].replace(" ", ""))






def taskC(tweets, components, output):
    tweet_map = {}

    for t in tweets:


        usr = t['user']['screen_name']
        text = t['text']
        if usr in tweet_map:
            tweet_map[usr] += ' ' + text
        else:
            tweet_map[usr] = text

        if 'retweeted_status' in t:

            rt = t['retweeted_status']['user']['screen_name']
                
            rtext = t['retweeted_status']['text']
            if rt in tweet_map:
                tweet_map[rt] += ' ' + rtext

            else:
                tweet_map[rt] = rtext
                   
    
    k_comm = components[-2:]
    rest = components[:-2]
    arr = []

    cluster = 0
    for k in k_comm:
        for usr in k:
            tweet = tweet_map[usr]
                
            bit = (usr, tweet, cluster)
            arr.append(bit)
        cluster +=1

    df = pd.DataFrame(arr, columns=['user', 'tweet', 'community'])
    train_len = len(df)

    arr = []

    for k in rest:
        for usr in k:
            tweet = tweet_map[usr]

            bit = (usr, tweet, -1)
            arr.append(bit)
    df2 = pd.DataFrame(arr, columns=['user', 'tweet', 'community'])
    df = df.append(df2)


    train = df[:train_len]
    test = df[train_len:]


    count_vect = CountVectorizer()

    features_train = train['tweet']
    labels_train = train['community']
    X_train_counts = count_vect.fit_transform(features_train)



    clf = MultinomialNB().fit(X_train_counts, labels_train)
    clf.score(X_train_counts, labels_train)

    X_new_counts = count_vect.transform(test['tweet'])

    #do the predictions
    preds = clf.predict(X_new_counts)


    test['community'] = preds
    train['community'] = clf.predict(X_train_counts)
    final = test.append(train)


    results_map = {}
    results_map[0] = []
    results_map[1] = []

    for index, row in final.iterrows():
        results_map[row['community']].append(row['user'])

    with open(output, "w", encoding="utf-8") as f:
        for k in sorted(results_map, key=lambda k: len(results_map[k]), reverse=False):
            result = results_map[k]
            push = sorted(result)
            f.write("%s\n" % str(push)[1:-1].replace(" ", ""))


if __name__ == "__main__":

   
    startTime = time.time()

    input_file = sys.argv[1]
    A_output = sys.argv[2]
    B_output = sys.argv[3]
    C_output = sys.argv[4]

    tweets = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            tweet = json.loads(line)

            tweets.append(tweet)

    components = taskA(tweets, A_output)


    taskB(tweets, components, B_output)


    taskC(tweets, components, C_output)

    executionTime = (time.time() - startTime)
  
    # print(executionTime)