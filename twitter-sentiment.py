import tweepy
import config
from datetime import date, timedelta

auth = tweepy.OAuthHandler(config.api_key, config.api_key_secret)
auth.set_access_token(config.access_token, config.access_token_secret)
api = tweepy.API(auth)

try:
    api.verify_credentials()
    print('Authentication OK')
except:
    print('Error during authentication')

