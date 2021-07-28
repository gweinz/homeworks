import sys
import pdb
import time
import pandas as pd
from sklearn.metrics import mean_squared_error
def get_usr_rat_dict(trainfile):
    ret_dict = {}
    df = pd.read_csv(trainfile)   

    for index, row in df.iterrows():

        movie = row['movieId']
        rating = row['rating']
        usr = row['userId']

        ret_dict[(usr, movie)] = rating 

    return ret_dict


def get_user_dict(trainfile):
    ret_dict = {}
    df = pd.read_csv(trainfile)   

    for index, row in df.iterrows():

        movie = row['movieId']
        rating = row['rating']
        usr = row['userId']

        if usr in ret_dict:
            ret_dict[usr].append(rating)

        else:
            ret_dict[usr] = [rating]

    for usr in ret_dict:
        ret_dict[usr] = sum(ret_dict[usr])/len(ret_dict[usr])

    return ret_dict

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def find_similar_list(shingles, query):
    sims = []

    for i, s in enumerate(shingles):
        sim = len(s & query) / len(s | query) # Jaccard Similarity
        if sim >= 0.1:
            sims.append((i, sim))

    return sorted(sims, key=lambda x: x[1], reverse=True)

def find_similar_from_list(shingles, query_index, threshold, sort=False):
    sims = [] #list of tuple
    query = shingles[query_index]

    for i, s in enumerate(shingles):
        sim = len(s & query) / len(s | query) # Jaccard Similarity
        if sim >= threshold:
            #tuple of (review index, similarity)
            sims.append((i, sim))

    if sort:
        #sort by similarity in decreasing order
        sims = sorted(sims, key=lambda x: x[1], reverse=True)

    return sims

def print_similar_items(sims, query_index, threshold):
    print(f'Reviews similiar to review #{query_index} with similarity greater than {threshold}')
    for sim in sims:
        print(sim)

    print(f'Total: {len(sims)}')

def shingle_dict(n_grams, filename):
    #db mappings each shingle to an unique index. shingle->index
    db = {}
    shingles = []

    for line in open(filename, 'r', encoding='utf-8'):
        tokens = [t for t in line[:-1].split(',') if  t != '']
        tokens = tokens[-1].split('|')
        shingle = set()
        for ngram in find_ngrams(tokens, n_grams):
            #frozenset to make it order independent and hashable
            ngram = frozenset(ngram)

            #same idea in week2, get_item_dict
            if ngram not in db:
                db[ngram] = len(db)

            shingle.add(db[ngram])
   
        shingles.append(shingle)

    return shingles, db



def get_recs(shingles, testfile, trainfile, moviefile, threshold, user_dict, usr_rat_dict):
    test_df = pd.read_csv(testfile)
    ratings_df = pd.read_csv(trainfile)
    movies_df = pd.read_csv(moviefile)
    nr = []

    # test_df['nr'] = test_df.apply(lambda row: find_similiar(bucket_list, matrix, M, row['movieId'], b, r, k_band, verify_by_large, user_dict, row['userId'], usr_rat_dict, movies_df), axis=1)
    for index, row in test_df.iterrows():

        user = row['userId']

        movieId = int(row['movieId'])
        query_index = movies_df.index[movies_df['movieId'] == movieId].tolist()[0]
        movie_name = movies_df[movies_df['movieId'] == movieId]['title']
        
        rts = []
        #find similar movies to said tested movie
        sims = find_similar_list(shingles, shingles[query_index])
        
       

        ct = 0
        for s in sims:
            #weighted avg
            if (user, s[0]) in usr_rat_dict:
                rt = usr_rat_dict[(user, s[0])]
                ct+=s[1]
                rts.append(rt)

        if len(rts) > 0:
            avg = sum(rts)/len(rts)
        else:
           
            avg = user_dict[user]

        a = round(avg, 1)
        nr.append(a)

     
   

    test_df['new_rating'] = nr
    return test_df

#python3 q2.py review/sample.txt 467 5 0.4
if __name__ == '__main__':
    filename = sys.argv[1]
    query_index = int(sys.argv[2])
    n_grams = int(sys.argv[3])
    threshold = float(sys.argv[4])

    testfile = 'ml-latest-small/ratings_test_truth.csv'
    time_start = time.time()
    trainfile= 'ml-latest-small/ratings_train.csv'
    # In text book the hash size is 4bytes = 32 bits = 2**32 = 4294967296
    # Since our data is small (10000 lines), 672391 shingles (from q1), 20 bits is enough. math.log2(672391)=19.35
    hash_size = 2**20


    user_dict = get_user_dict(trainfile)

    usd = get_usr_rat_dict(trainfile)

    # shingles, _ = get_shingle_hash(n_grams, hash_size, filename)
    shingles, db = shingle_dict(n_grams, filename)
    scores = get_recs(shingles, testfile, trainfile, filename, threshold, user_dict, usd)
    scores.to_csv('check.csv')


    y_true = scores['rating']
    y_pred = scores['new_rating']

    print("MSE of: ",mean_squared_error(y_true, y_pred))

    # sims = find_similar_from_list(shingles, query_index, threshold, True)

    # print_similar_items(sims, query_index, threshold)

    # time_end = time.time()
    # elapsed = time_end - time_start
    # print(f'Elapsed: {elapsed}')
