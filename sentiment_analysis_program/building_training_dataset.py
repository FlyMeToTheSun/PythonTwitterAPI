import twitter
import tweepy
#import python-twitter

import nltk
import re
from nltk.corpus import stopwords,wordnet
import unicodedata
import string
from string import punctuation
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer

twitter_API = twitter.Api(consumer_key='5HwCJo0YmNDrtagqmwoz5DQkg',
                      consumer_secret='R9ZWO7FZtXVWG6kjhHBp8zqLPeaKYZG9XAFxFBgHHyyqDSBUz1',
                      access_token_key='1534060333777575936-hqzIO99GZKpMNibBLlHuBwsgUf4PBf',
                      access_token_secret='MuSZpaTkHxFuAvsL8WtVQfkRGPSMPJeI7rIsbnZwfIaCJ')

def buildTrainingSet(corpusFile, tweetDataFile):
    import csv
    import time

    corpus = []

    with open(corpusFile,'rt') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2], "label":row[1], "topic":row[0]})

    rate_limit = 180
    sleep_time = 1

    #rate_limit = 10
    #sleep_time = 20/10

    trainingDataSet = []

    for tweet in corpus:
        try:
            status = twitter_API.GetStatus(tweet["tweet_id"])
            print("Tweet fetched" + status.text)
            tweet["text"] = status.text
            trainingDataSet.append(tweet)
            time.sleep(sleep_time)
        except Exception as e:
            print(e)
            continue
    # now we write them to the empty CSV file
    with open(tweetDataFile,'w', encoding="utf-8") as csvfile:
        linewriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
        rows ="tweet_id,text,label,topic"
        linewriter.writerow(rows)
        try:
                linewriter.writerow([tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
            except Exception as e:
                print(e)
        print("successful")
    return trainingDataSet

corpusFile = "output/corpus-100.csv"
tweetDataFile = "output/tweetDataFile-text.csv"

trainingDataSet = buildTrainingSet(corpusFile, tweetDataFile)
