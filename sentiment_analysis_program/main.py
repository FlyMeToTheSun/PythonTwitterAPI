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

# BUILDING TRAINING SET



# Pre-processing Tweets in The Data Sets

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

    def processTrainingSet(self, list_of_tweets):
        processedTweets=[]
        file = open(list_of_tweets, 'r', encoding="utf-8",)
        lines = file.readlines()
        count = 0
        for index, line in enumerate(lines):
            if line.strip():
                stripped_line = (line.strip())
                stripped_line = list(stripped_line.split(","))
                try:
                    processedTweets.append((self._processTweet(stripped_line[1]), stripped_line[2]))
                except:
                    print("Missing Values on Line: ",count)
                count +=1
        return processedTweets

    def processTestSet(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets


    def _processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # Remove URL
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames Target Mentions, & re-tweets,
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag & Hashtags,
        tweet = re.sub(r'[^\w\s]','',tweet)          # Remove special characters and punctuation
        tweet = re.sub(r"([0-9])", r" ",tweet)       # Remove Numerical data
        tweet = re.sub("(.)\\1{2,}", "\\1", tweet)   # Remove duplicate characters
        tweet = word_tokenize(tweet) # Tokenization & Remove repeated characters
        #tweet_result = [word for word in tweet if word not in self._stopwords]
        #return tweet_result
        lemmatizated_words = [word for word in tweet if word not in self._stopwords]
        lemmatizated_words = [self.lemmatizer.lemmatize(tweet,self.get_wordnet_pos(tweet)) for tweet in lemmatizated_words] # Lemmatization;
        lemmatizated_words = [self.porter.stem(word) for word in lemmatizated_words]   # Stemming;
        lemmatizated_words = [self.lancaster.stem(word) for word in lemmatizated_words] #  Stemming;
        return lemmatizated_words

trainingDataSetLink = 'output/tweetDataFile.csv'

tweetProcessor = PreProcessTweets()

preprocessedTrainingSet = tweetProcessor.processTrainingSet(trainingDataSetLink)
preprocessedTestSet = tweetProcessor.processTestSet(testDataSet)
print(preprocessedTrainingSet)
print(preprocessedTestSet)


#with open('output/preprocessedTestSetFile.txt', 'w', encoding="utf-8") as f:
#    f.write(str(preprocessedTestSet))

#with open('output/preprocessedTrainingSetFile.txt', 'w', encoding="utf-8") as f:
#    f.write(str(preprocessedTrainingSet))


# Naive Bayes Classifier
import nltk
print("Beginining Naive Bayes Classifier")
def buildVocabulary(preprocessedTrainingData):
    all_words = []

    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    print("Vocabulary Built")
    return word_features

def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)

    print("Features Extracted")
    return features


# Now we can extract the features and train the classifier
word_features = buildVocabulary(preprocessedTrainingSet)
trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)
print("Currently Training...")
NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
print("Training Finished")
print("Currently Classifying...")
NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]
print("Classifying Finished")
# get the majority vote
if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
else:
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")

print("Overall Positive Sentiment")
print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
print("Overall Negative Sentiment")
print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
