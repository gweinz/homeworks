import pandas as pd 
import numpy as np
import sys



def get_avg_ratings(trainfile):
    ret_dict = {}
    df = pd.read_csv(trainfile) 

    for index, row in df.iterrows():

        movie = int(row['movieId'])
        rating = row['rating']
        usr = int(row['userId'])
        if usr in ret_dict:
            ret_dict[usr-1].append(rating)
        else:
            ret_dict[usr-1] = [rating]

    for r in ret_dict:
        ret_dict[r] = sum(ret_dict[r])/len(ret_dict[r])
    return ret_dict

def get_usr_rat_dict(trainfile):
    ret_dict = {}
    df = pd.read_csv(trainfile)   

    for index, row in df.iterrows():

        movie = int(row['movieId'])
        rating = row['rating']
        usr = int(row['userId'])

        ret_dict[(usr, movie)] = rating 

    return ret_dict


def get_movie_avg_ratings(trainfile):
    ret_dict = {}
    df = pd.read_csv(trainfile) 

    for index, row in df.iterrows():

        movie = int(row['movieId'])
        rating = row['rating']
        usr = int(row['userId'])
        if movie in ret_dict:
            ret_dict[movie].append(rating)
        else:
            ret_dict[movie] = [rating]

    for r in ret_dict:
        ret_dict[r] = sum(ret_dict[r])/len(ret_dict[r])
        
    return ret_dict

if __name__ == '__main__':
    moviefile = sys.argv[1]
    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    outfile = sys.argv[4]


    df_ratings = pd.read_csv(trainfile)
    df = df_ratings.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    m_avgs = get_movie_avg_ratings(trainfile)
    u_avgs = get_avg_ratings(trainfile)
    global_avgs = df.replace(0, np.NaN).mean()

    g_avg = sum(global_avgs)/len(global_avgs)

    df = df.replace(0, np.NaN)


    for col in df.columns:
        df[col] =df[col].fillna(df[col].mean())

    mov_map = {}
    idx_map = {}
    i = 0

    list_movies = df_ratings['movieId'].unique()
    c = len(list_movies)

    for mv in list_movies:
        mov_map[mv] = i 
        idx_map[i] = mv
        i+=1


    X = np.array(df)
    rows = X.shape[0]
    cols = X.shape[1]

    for i in range(rows):
        u_a = u_avgs[i]
        for j in range(cols):
          
            movie = idx_map[j]
            
            m_a = m_avgs[movie]
            m_bias = m_a - g_avg
            u_bias = u_a - g_avg


            X[i][j] += m_bias
            X[i][j] += u_bias



    X = np.transpose(X)

    u, s, vh = np.linalg.svd(X)


    s = np.sqrt(s)
    s_u = np.zeros(X.shape)
    for i, v in enumerate(s):
        s_u[i, i] = v

    s_v = np.diag(s)

    r = 10

    A = np.matmul(u, s_u)[:, :r]

    B = np.matmul(s_v, vh)[:r, :]
    AB = np.matmul(A, B)
    loss = np.mean((AB - X)**2)



    R = np.transpose(AB)

    for i in range(rows):
        u_a = u_avgs[i]
        for j in range(cols):
          
            movie = idx_map[j]
            
            m_a = m_avgs[movie]
            m_bias = m_a - g_avg
            u_bias = u_a - g_avg


            R[i][j] -= m_bias
            R[i][j] -= u_bias

            if R[i][j] > 5:
                R[i][j] = 5

            if R[i][j] < 0:
                R[i][j] = 0

     
    out = pd.read_csv(testfile)
    preds = []
    for index, row in out.iterrows():
        movieId = int(row['movieId'])
        usr = int(row['userId'])
        if movieId in mov_map:

            movie_idx = int(mov_map[movieId])

            val = R[usr-1][movie_idx]
            preds.append(val)

        else:
            preds.append(g_avg)

 
    out['rating'] = preds
    out.to_csv(outfile, index=False)
   


