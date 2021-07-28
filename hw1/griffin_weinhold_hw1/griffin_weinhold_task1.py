from pyspark import SparkContext
import json
import sys

def main(file):
    responses = {}
    sc = SparkContext("local[*]","PySpark Tutorial")

    lines = sc.textFile(file)
    n_tweet = lines.count()
    responses['n_tweet'] = n_tweet

    lineJSONS = lines.map(lambda s: (json.loads(s)))
    userJSONS = lineJSONS.map(lambda x: x['user'])

    users = lineJSONS.map(lambda x: x['user']['id'])
    n_user = users.distinct().count()
    responses['n_user'] = n_user

    follows = userJSONS.map(lambda x: (x['screen_name'], x['followers_count']))
    popular_users = follows.sortBy(lambda x: x[1], ascending=False).take(3)
    responses['popular_users'] = popular_users

    tuesdays = lineJSONS.filter(lambda x: x['created_at'][:3] == 'Tue')
    Tuesday_Tweet = tuesdays.count()
    responses['Tuesday_Tweet'] = Tuesday_Tweet

    return responses

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]

    results = main(infile)

    with open(outfile, 'w') as fout:
        json.dump(results, fout)