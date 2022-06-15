import twitter
import tweepy

import nltk
import re
from nltk.corpus import stopwords,wordnet
import unicodedata
import string
from string import punctuation
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
from os.path import exists

import csv

# INIT Twitter API
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

testDataFile = "testDataFile.csv"
testDataSet = buildTestSet("apple")


# BUILDING TRAINING SET
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

corpusFile = "corpus.csv"
tweetDataFile = "tweetDataFile.csv"

file_exists = exists(tweetDataFile)
if file_exists == False:
    trainingDataSet = buildTrainingSet(corpusFile, tweetDataFile)
else:
    print("Using previously saved data as Training Set")


# PRE-PROCESSING TRAINING AND TEST SETS

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL']) # Stop words removal

    porter = PorterStemmer()
    lancaster=LancasterStemmer()
    lemmatizer = WordNetLemmatizer()

    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag,wordnet.NOUN)

    def processTweet(self, list_of_tweets):
        processedTweets=[]
        file = open(list_of_tweets, 'r', encoding="utf-8",)
        lines = file.readlines()
        count = 0
        for index, line in enumerate(lines):
            if line.strip():
                stripped_line = (line.strip())
                stripped_line = list(stripped_line.split(","))
                try:
                    if  ((stripped_line[2] != "neutral") and (stripped_line[2] != "irrelevant")):
                        processedTweets.append((self._processTweet(stripped_line[1]), stripped_line[2]))
                except:
                    print("Missing Training Set Values on Line: ",count)
                count +=1
        return processedTweets

    def _processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # Remove URL
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # Remove usernames Target Mentions, & re-tweets,
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # Remove the # in #hashtag & Hashtags,
        tweet = re.sub(r'[^\w\s]','',tweet)          # Remove special characters and punctuation
        tweet = re.sub(r"([0-9])", r" ",tweet)       # Remove Numerical data
        tweet = re.sub("(.)\\1{2,}", "\\1", tweet)   # Remove duplicate characters
        tweet = word_tokenize(tweet) # Tokenization & Remove repeated characters
        tweet_result = [word for word in tweet if word not in self._stopwords]
        lemmatizated_words = [word for word in tweet if word not in self._stopwords]
        lemmatizated_words = [self.lemmatizer.lemmatize(tweet,self.get_wordnet_pos(tweet)) for tweet in lemmatizated_words] # Lemmatization;
        lemmatizated_words = [self.porter.stem(word) for word in lemmatizated_words]   # Stemming;
        lemmatizated_words = [self.lancaster.stem(word) for word in lemmatizated_words] #  Stemming;
        return lemmatizated_words

trainingDataFile = 'tweetDataFile.csv'

tweetProcessor = PreProcessTweets()
preprocessedTestSet = tweetProcessor.processTweet(testDataFile)
preprocessedTrainingSet = tweetProcessor.processTweet(trainingDataFile)

print("preprocessedTrainingSet = ",preprocessedTrainingSet)
print("preprocessedTestSet = ", preprocessedTestSet)

# NAIVE BAYES CLASSIFIER FOR SENTIMENT ANALYSIS
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

word_features = buildVocabulary(preprocessedTrainingSet)
print("word_features = ", word_features)
trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)
print("trainingFeatures = ", trainingFeatures)


NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
print("NBayesClassifier = ", NBayesClassifier)


NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]
print("NBResultLabels = ", NBResultLabels)
print("\n")
print("Classification Complete")
print("=======================")
# get the majority vote
if ((NBResultLabels.count('positive') > NBResultLabels.count('negative')) and (NBResultLabels.count('positive') > NBResultLabels.count('neutral'))):
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
elif ((NBResultLabels.count('positive') < NBResultLabels.count('negative')) and (NBResultLabels.count('negative') > NBResultLabels.count('neutral'))):
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
elif ((NBResultLabels.count('positive') < NBResultLabels.count('neutral')) and (NBResultLabels.count('negative') < NBResultLabels.count('neutral'))):
    print("Overall Neutral Sentiment")

else:
    print("Overall Irrelevant Sentiment")


print("\n")
print("Sentiment Scores: Positive =" + str(NBResultLabels.count('positive')/len(NBResultLabels))+". Negative = " + str(NBResultLabels.count('negative')/len(NBResultLabels))+".")

#print("Neutral Sentiment Percentage = " + str(100*NBResultLabels.count('neutral')/len(NBResultLabels)) + "%")
#print("Irrelevant Sentiment Percentage = " + str(100*NBResultLabels.count('irrelevant')/len(NBResultLabels)) + "%")
