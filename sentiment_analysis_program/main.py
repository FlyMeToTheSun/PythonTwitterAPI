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

import csv

# initialize api instance
twitter_API = twitter.Api(consumer_key='5HwCJo0YmNDrtagqmwoz5DQkg',
                      consumer_secret='R9ZWO7FZtXVWG6kjhHBp8zqLPeaKYZG9XAFxFBgHHyyqDSBUz1',
                      access_token_key='1534060333777575936-hqzIO99GZKpMNibBLlHuBwsgUf4PBf',
                      access_token_secret='MuSZpaTkHxFuAvsL8WtVQfkRGPSMPJeI7rIsbnZwfIaCJ')

# BUILDING TEST SET

def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_API.GetSearch(search_keyword, count = 20)
        dataLine = [{"text":status.text, "label":None} for status in tweets_fetched]
        with open(testDataFile,'w', encoding="utf-8") as csvfile:
            linewriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
            try:
                linewriter.writerow({"text":status.text, "label":None} for status in tweets_fetched)
            except Exception as e:
                print(e)
            print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
        return dataLine
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print (message)
        return None

testDataFile = "output/testDataFile.csv"
testDataSet = buildTestSet("smile")



#print(testDataSet[0:4])


# BUILDING TRAINING SET



# Pre-processing Tweets in The Data Sets
import json
import ast
class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    porter = PorterStemmer()
    lancaster=LancasterStemmer()
    lemmatizer = WordNetLemmatizer()

    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag,wordnet.NOUN)

    def processTweets(self, list_of_tweets):
        processedTweets=[]
        #for line in list_of_tweets:
        #    processedTweets.append((self._processTweet(line['text']), line['label']))

        file = open('output/tweetDataFile-100.csv', 'r', encoding="utf-8",)
        lines = file.readlines()

        for index, line in enumerate(lines):
            if line.strip():
                stripped_line = (line.strip())
                stripped_line = list(stripped_line.split(","))

                processedTweets.append((self._processTweet(stripped_line[1]), stripped_line[2]))
                print(stripped_line)
        print(processedTweets)
        return processedTweets


    def _processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        #tweet_result = [word for word in tweet if word not in self._stopwords]
        #return tweet_result
        return [word for word in tweet if word not in self._stopwords]

with open('output/tweetDataFile-100.txt', 'r', encoding="utf-8",) as f:
    trainingDataSet = f.read( )

print(trainingDataSet)

tweetProcessor = PreProcessTweets()
preprocessedTestSetFile = "output/preprocessedTestSetFile.csv"
preprocessedTrainingSetFile = "output/preprocessedTrainingSetFile.csv"
#print(testDataSet)
#print(trainingDataSet)

preprocessedTestSet = tweetProcessor.processTweets(trainingDataSet)
#preprocessedTestSet = tweetProcessor.processTweets(testDataSet)
preprocessedTestSet = str(preprocessedTestSet)

#with open('output/preprocessedTestSetFile.txt', 'w', encoding="utf-8") as f:
#    f.write(preprocessedTestSet)

"""
# Naive Bayes Classifier
import nltk

def buildVocabulary(preprocessedTrainingData):
    all_words = []

    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()

    return word_features

def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
    return features


# Now we can extract the features and train the classifier
word_features = buildVocabulary(preprocessedTrainingSet)
trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)


NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]

# get the majority vote
if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
else:
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
"""
