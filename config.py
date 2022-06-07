import os
import json
import tweepy
from tweepy import Stream                   # Useful in Step 3
from tweepy.streaming import StreamListener # Useful in Step 3
consumer_key = os.getenv(“CONSUMER_KEY_TWITTER”)
consumer_secret = os.getenv(“CONSUMER_SECRET_TWITTER”)
access_token = os.getenv(“ACCESS_KEY_TWITTER”)
access_token_secret = os.getenv(“ACCESS_SECRET_TWITTER”)
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
