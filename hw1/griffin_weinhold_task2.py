from pyspark import SparkContext
import json
import sys


def main(file):
    responses = {}
    sc = SparkContext("local[*]","PySpark Tutorial")

    lines = sc.textFile(file)
    lineJSONS = lines.map(lambda s: (json.loads(s)))
    rts = lineJSONS.map(lambda x: x['retweet_count'])

    mean_retweet = rts.mean()
    responses['mean_retweet'] = mean_retweet

    max_retweet = rts.max()
    responses['max_retweet'] = max_retweet

    
    stdev_retweet = rts.stdev()
    responses['stdev_retweet'] = stdev_retweet


    return responses

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]

    results = main(infile)

    with open(outfile, 'w') as fout:
        json.dump(results, fout)