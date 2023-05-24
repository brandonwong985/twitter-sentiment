import tweepy
import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

KEYWORD = 'Kim Chaewon'
USERNAME = 'bobawastedbot'
TWEET_COUNT = 10

auth = tweepy.OAuthHandler(config.api_key, config.api_key_secret)
auth.set_access_token(config.access_token, config.access_token_secret)
api = tweepy.API(auth)

try:
    api.verify_credentials()
    print('Authentication OK')
except:
    print('Error during authentication')

roberta = 'cardiffnlp/twitter-roberta-base-sentiment'
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

tweets = api.search_tweets(q=KEYWORD, tweet_mode = 'extended', count = TWEET_COUNT)
ids = []
sentiment = []
for t in tweets:
    words = []
    ids.append(t.id)
    #print(t)
    for word in t.full_text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        elif word.startswith('#') and len(word) > 1:
            word = '#hashtag'
        words.append(word)
    proccessed_tweet = ' '.join(words)
    encoded_tweet = tokenizer(proccessed_tweet, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = list(softmax(scores))
    res = scores.index(max(scores))
    sentiment.append(res)
    print(f'tweet: {t.id} \n{proccessed_tweet}\n   {res} {labels[res]}')
    
print(f'neg: {sentiment.count(0)}, neu: {sentiment.count(1)}, pos: {sentiment.count(2)}')
msg = f'This is what 10 recent Tweeters think about {KEYWORD}:\nNegative: {sentiment.count(0)}\nNeutral: {sentiment.count(1)}\nPositive: {sentiment.count(2)}'
api.update_status(status=msg)

my_tweet = api.user_timeline(screen_name=USERNAME, count=1, exclude_replies=True, include_rts=False)
reply = 'Tweets sourced:\n'
for id in ids[:]:
    reply += f'{id}\n'
api.update_status(status=reply, in_reply_to_status_id = my_tweet[0].id)