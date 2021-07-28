from pyspark import SparkContext
import json
import sys


def main(file):
    responses = {}
    sc = SparkContext("local[*]","PySpark Tutorial")
    tweetlines = sc.textFile(file)

    tweets = tweetlines.flatMap(lambda x: x.split(' '))
    keys = tweets.map(lambda x: (x,1)).reduceByKey(lambda a, b: a+b)
    max_word = keys.map(lambda x: (x[1], x[0])).sortByKey(False).take(1)
    responses['max_word'] = [max_word[0][1], max_word[0][0]]

    mindless = keys.filter(lambda x: x[0] == 'mindless').collect()
    mindless_count = mindless[0][1]
    responses['mindless_count'] = mindless_count

    chunks = keys.filter(lambda x: x[0] == '|********************').take(1)
    chunk_count = chunks[0][1]
    responses['chunk_count'] = chunk_count

    return responses

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]

    results = main(infile)

    with open(outfile, 'w') as fout:
        json.dump(results, fout)