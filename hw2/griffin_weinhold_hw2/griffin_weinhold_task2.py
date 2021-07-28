import sys
import numpy as np
import collections
import pandas as pd


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def get_shingle_hash(n_grams, hash_size, filename):
    # The hash table could be stored as a sparse matrix. However, I did a few benchmark, they are not as efficient as store non-zero entries in a list
    # So keep use the same layout as in q1
    shingles = [] #list of set

    for line in open(filename, 'r', encoding='utf-8'):
        tokens = [t for t in line[:-1].split(',') if  t != '']
        tokens = tokens[-1].split('|')
        shingle = set()
        for ngram in find_ngrams(tokens, n_grams):
            #frozenset to make it order independent and hashable
            ngram = frozenset(ngram)

            ngram_hash = hash(ngram) % hash_size

            shingle.add(ngram_hash)
        shingles.append(shingle)

    # return None to keep the same signature as q1
    return shingles, None

def print_similar_items(sims, query_index, threshold):
    print(f'Reviews similiar to review #{query_index} with similarity greater than {threshold}')
    for sim in sims:
        print(sim)

    print(f'Total: {len(sims)}')

def get_hash_coeffs(br):
    #hash(x) = (a*x + b) % c
    #a, b are random integers less than maximum value of x.
    #Here I choose a, b in range [0, 2**10). Because x is in range [0, 2**20), this choice of a, b keep a*x+b inside the range of int32 [0, 2**32), be more efficient. In python you don't need to worry about overflow, in language like C, exceeding the range of int32 will cause overflow.
    #c is a prime number greater than 2**20. Look it up from http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
    rnds = np.random.choice(2**10, (2, br), replace=False)
    c = 1048583
    return rnds[0], rnds[1], c

def min_hashing(shingles, hash_coeffs, br):
    count = len(shingles)

    (a, b, c) = hash_coeffs
    a = a.reshape(1, -1)
    M = np.zeros((br, count), dtype=int) #Its layout same as slide 56. col are docs, row are signature index
    for i, s in enumerate(shingles):
        # All shingles in the document
        row_idx = np.asarray(list(s)).reshape(-1, 1)
        # Instead of getting many hash functions and run each hash function to each shingles,
        # Use numpy matrix multiplication to apply all hash funcitons to all shingles in the same time
        m = (np.matmul(row_idx, a) + b) % c
        m_min = np.min(m, axis=0) #For each hash function, minimum hash value for all shingles
        M[:, i] = m_min

    return M

def LSH(M, b, r, band_hash_size):
    count = M.shape[1]

    bucket_list = []
    for band_index in range(b):
        # The hash table for each band is stored as a dictionrary of sets. It's more efficient than sparse matrix
        m = collections.defaultdict(set)

        row_start = band_index * r
        for c in range(count):
            v = M[row_start:(row_start+r), c]
            v_hash = hash(tuple(v.tolist())) % band_hash_size
            m[v_hash].add(c)

        bucket_list.append(m)

    return bucket_list


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

def get_recs(trainfile, moviefile, testfile, bucket_list, shingles, M, b, r, band_hash_size, verify_by_signature, user_dict, usr_rat_dict, threshold):
    ratings_df = pd.read_csv(trainfile)
    movies_df = pd.read_csv(moviefile)
    test_df = pd.read_csv(testfile)
    nr = []

    # test_df['nr'] = test_df.apply(lambda row: find_similiar(bucket_list, matrix, M, row['movieId'], b, r, k_band, verify_by_large, user_dict, row['userId'], usr_rat_dict, movies_df), axis=1)
    for index, row in test_df.iterrows():

        user = row['userId']

        movieId = int(row['movieId'])
        query_index = movies_df.index[movies_df['movieId'] == movieId].tolist()[0]
        movie_name = movies_df[movies_df['movieId'] == movieId]['title']
        
        rts = []
        #find similar movies to said tested movie
        sims = find_similiar(shingles, query_index, threshold, bucket_list, M, b, r, band_hash_size, verify_by_signature)


      

        ct = 0
        for s in sims:
            #weighted avg
            if (user, s[0]) in usr_rat_dict:
                rt = s[1]*usr_rat_dict[(user, s[0])]
                ct+=s[1]
                rts.append(rt)

        if len(rts) > 0:
            avg = sum(rts)/ct
        else:
      
            
            avg = user_dict[user]

        # a = round(avg, 1)
        nr.append(avg)

     
   
    test_df['rating'] = nr
    # test_df['new_rating'] = nr
    return test_df
        
def find_similiar(shingles, query_index, threshold, bucket_list, M, b, r, band_hash_size, verify_by_signature):
    # Step 1: Find candidates
    candidates = set()
    for band_index in range(b):
        row_start = band_index * r
        v = M[row_start:(row_start+r), query_index]
        v_hash = hash(tuple(v.tolist())) % band_hash_size

        m = bucket_list[band_index]
        bucket = m[v_hash]
        # print(f'Band: {band_index}, candidates: {bucket}')
        candidates = candidates.union(bucket)

    # print(f'Found {len(candidates)} candidates')

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

    

#python3 q3.py review/sample.txt 467 5 0.4
if __name__ == '__main__':
    moviefile = sys.argv[1]
    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    outfile = sys.argv[4]

    
    n_grams = 1
    threshold =.1

    hash_size = 2**20      # hashtable size for k-shingle
    band_hash_size = 2**20 # hashtable size for each band, could use smaller bucket size
    verify_by_signature = False

    #b, r are not choose arbitrarily. (1/b)^(1/r) should be close to the threshold we chose. See slide 106
    b = 70
    r = 2
    br = b*r

 

    # same as q2
    shingles, _ = get_shingle_hash(n_grams, hash_size, moviefile)

    # Instead of getting many hash functions and run each hash function to each shingles,
    # Use numpy matrix multiplication to apply all hash funcitons to all shingles in the same time
    hash_coeffs = get_hash_coeffs(br)

    M = min_hashing(shingles, hash_coeffs, br) #col are docs, row are signature index
    bucket_list = LSH(M, b, r, band_hash_size) #list of sparse matrix



    user_dict = get_user_dict(trainfile)

    usd = get_usr_rat_dict(trainfile)



    out = get_recs(trainfile, moviefile, testfile, bucket_list, shingles, M, b, r, band_hash_size, verify_by_signature, user_dict, usd, threshold)
    
  
    out.to_csv(outfile,index=False)

