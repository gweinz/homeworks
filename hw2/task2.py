from collections import Counter
import math
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy
import time
import itertools
import sys
from sklearn.metrics import mean_squared_error

ratings = pd.read_csv('ml-latest-small/ratings_train.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

def get_hash_config(k_shingle):
    rnds = np.random.choice(2**10, (2, k_shingle), replace=False)
    c = 1048583
    return rnds[0], rnds[1], c

def min_hashing(matrix, hash_config, k_shingle):
    count = matrix.shape[1]
   
    (a, b, c) = hash_config
    a = a.reshape(1, -1)
    M = np.zeros((k_shingle, count), dtype=int)
    for i in range(count):
        row_idx = matrix[:, i].indices.reshape(-1, 1)
      
        m = (np.matmul(row_idx, a) + b) % c
        if len(m)>0:
            m_min = np.min(m, axis=0)
            M[:, i] = m_min

    return M

def LSH(M, b, r, k_band):
    lines = M.shape[1]

    bucket_list = []
    for band_index in range(b):
        row_idx = []
        col_idx = []

        row_start = band_index * r
        for c in range(lines):
            v = M[row_start:(row_start+r), c]
            v_hash = hash(tuple(v.tolist())) % k_band
            row_idx.append(v_hash)
            col_idx.append(c)
            #m[v_hash, c] = 1
        data_ary = [True] * len(row_idx)

        m = scipy.sparse.csr_matrix((data_ary, (row_idx, col_idx)), shape=(k_band, lines), dtype=bool)
        bucket_list.append(m)
  
    return bucket_list
def og_similiar(bucket_list, matrix, M, query_index, b, r, k_band, verify_by_large):
    candidates = set()
    for band_index in range(b):
        row_start = band_index * r
        v = M[row_start:(row_start+r), query_index]
        v_hash = hash(tuple(v.tolist())) % k_band

        m = bucket_list[band_index]
        # print(f'Band: {band_index}, candidates: {m[v_hash].indices}')
        candidates = candidates.union(m[v_hash].indices)

    # print(f'Found {len(candidates)} candidates')

    sims = []
    # Since the candidates size is small, we just evaluate it on oringal matrix
    # Although you could do it on the signature matrix by checking how many elements are the same

    if verify_by_large:
        query_set = set(matrix[:, query_index].indices)
        for col_idx in candidates:
            col = matrix[:, col_idx]
            col_set = set(col.indices)

            sim = len(query_set & col_set) / len(query_set | col_set)
            if sim >= 0.7:
              

                sims.append((col_idx, sim))

    else:
        query_vec = M[:, query_index]
        for col_idx in candidates:
            col = M[:, col_idx]
            sim = np.mean(col == query_vec)
            if sim >= 0.7:

                sims.append((col_idx, sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims


def find_similiar(shingles, query_index, threshold, bucket_list, M, b, r, band_hash_size, verify_by_signature):
    # Step 1: Find candidates
    candidates = set()
    for band_index in range(b):
        row_start = band_index * r
        v = M[row_start:(row_start+r), query_index]
        v_hash = hash(tuple(v.tolist())) % band_hash_size

        m = bucket_list[band_index]
        bucket = m[v_hash]
        print(f'Band: {band_index}, candidates: {bucket}')
        candidates = candidates.union(bucket)

    print(f'Found {len(candidates)} candidates')

    # Step 2: Verify similarity of candidates
    sims = []
    # Since the candidates size is small, we just evaluate it on k-shingles matrix, or signature matrix for greater efficiency
    if verify_by_signature:
        query_vec = M[:, query_index]
        for col_idx in candidates:
            col = M[:, col_idx]
            sim = np.mean(col == query_vec) # Jaccard Similarity is proportional to the fraction of the minhashing signature they agree
            if sim >= threshold:
                sims.append((col_idx, sim))
    else:
        query_set = shingles[query_index]
        for col_idx in candidates:
            col_set = shingles[col_idx]

            sim = len(query_set & col_set) / len(query_set | col_set) # Jaccard Similarity
            if sim >= threshold:
                sims.append((col_idx, sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims



def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def shingle_hash(n_grams, hash_size, filename):

    row_idx = []
    col_idx = []
    lines = 0
    df = pd.read_csv(filename)
    for i, line in enumerate(open(filename, 'r', encoding='utf-8')):
        lines += 1
        tokens = [t for t in line[:-1].split(',') if  t != '']
    # for i, row in df.iterrows():

       
        tokens = tokens[-1].split('|')

        ngram = frozenset(tokens)
        # print(ngram)
        ngram_hash = hash(ngram) % hash_size
        
        # print(ngram_hash)
        row_idx.append(ngram_hash)
        col_idx.append(i)
        # for ngram in find_ngrams(tokens, n_grams):
        #     #frozenset to make it order independent and hashable
           
        #     ngram = frozenset(ngram)
        #     print(ngram)
            
        #     ngram_hash = hash(ngram) % hash_size
        
        #     print(ngram_hash)
        #     row_idx.append(ngram_hash)
        #     col_idx.append(i)
        


    data_ary = [True] * len(row_idx)

    # Convert to column based matrix for fast processing
   
    to_return = scipy.sparse.csc_matrix((data_ary, (row_idx, col_idx)), shape=(hash_size, lines), dtype=bool)
    

    return to_return

def get_rating_dict(trainfile):
    ret_dict = {}
    train_df = pd.read_csv(trainfile)   

    for index, row in train_df.iterrows():
        movie = row['movieId']
        rating = row['rating']
        if movie in ret_dict:
            ret_dict[movie].append(rating)
        else:
            ret_dict[movie] = [rating]
    for m in ret_dict:
        ret_dict[m] = (sum(ret_dict[m])/len(ret_dict[m]))

    return ret_dict

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

def get_recs(testfile, trainfile, moviefile, bucket_list, matrix, M, b, r, k_band, verify_by_large, user_dict, usr_rat_dict):
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
        sims = og_similiar(bucket_list, matrix, M, query_index, b, r, k_band, verify_by_large)
        
        print(index)
        # print("Movie",movie_name)
        # print(index)
        # print(movieId)
        # print(query_index)
        # print(sims)


        ct = 0
        for s in sims:
            if (user, s[0]) in usr_rat_dict:
                rt = s[1] * usr_rat_dict[(user, s[0])]
                ct+=s[1]
                rts.append(rt)

        if len(rts) > 0:
            avg = sum(rts)/ct
        else:
            avg = user_dict[user]
      


        
    
  
        nr.append(avg)

     
   

    test_df['new_rating'] = nr
    return test_df
        
        
    

if __name__ == '__main__':
    moviefile = sys.argv[1]
    trainfile = sys.argv[2]
    testfile = sys.argv[3]

    n_grams = 1
    hash_size = 2**16

    b = 30
    r = 3
    k_shingle = b * r
    k_band = 2**16
    verify_by_large = False

    matrix = shingle_hash(n_grams, hash_size, moviefile)
    
  
  
    hash_config = get_hash_config(k_shingle)
    M = min_hashing(matrix, hash_config, k_shingle)

    bucket_list = LSH(M, b, r, k_band)

    # sims = find_similiar(bucket_list, matrix, M, 1400, b, r, k_band, verify_by_large)
    # print('Index of similiar review with similarity greater than 0.4')
    # print(sims)
    user_dict = get_user_dict(trainfile)

    usd = get_usr_rat_dict(trainfile)

    scores = get_recs(testfile, trainfile, moviefile, bucket_list, matrix, M, b, r, k_band, verify_by_large, user_dict, usd)
    scores.to_csv('check.csv')


    y_true = scores['rating']
    y_pred = scores['new_rating']

    print("MSE of: ",mean_squared_error(y_true, y_pred))
   
