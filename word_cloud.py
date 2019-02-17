''' 
    Word cloud maker for twitter
    Input twitter screen name and get a word cloud from their latest tweets
    Latest tweets include 3200 tweets including retweets from their timeline
    This is meant to use for tweets written in English
'''

from api_keys import API_KEY, API_SECRET
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from twython import Twython
from wordcloud import WordCloud

import matplotlib.pyplot as plt


twitter = Twython(API_KEY, API_SECRET)
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
english_stops = set(stopwords.words('english'))
wnet_lemmatzr = WordNetLemmatizer()

while True:
    user = input("Twitter screen name: ")
    
    # Get the latest tweet from the user and the id of it
    user_timeline = twitter.get_user_timeline(screen_name=user, count=1) 
    try:
        last_tweet_id = user_timeline[0]['id']-1
    except IndexError:
        print("Couldn't find any tweets from {}".format(user))
        continue
    
    # You can only get 200 tweets at once and 3200 tweets altogether with free tier API access
    # So we need to get 200 tweets at a time and use the oldest tweet's id as a limiter for the next batch of 200
    for i in range(16):
        tweets = twitter.get_user_timeline(screen_name=user, count=200, max_id=last_tweet_id)
        user_timeline.extend(tweets)
        last_tweet_id = user_timeline[-1]['id'] - 1
    
    # Text preprocessing    
    raw_tweets = [tweet['text'] for tweet in user_timeline]    
    tweet_tokens = [tknzr.tokenize(tweet) for tweet in raw_tweets]
    lower_tokens = []
    
    for tweet in tweet_tokens:
        for token in tweet:
            token = token.lower()
            if not token.startswith('http'):
                lower_tokens.append(token)
    
    alpha_tokens = [token for token in lower_tokens if token.isalpha()]
    stops_removed = [token for token in alpha_tokens if token not in english_stops]
    lemmatized_tokens = [wnet_lemmatzr.lemmatize(token) for token in stops_removed if len(token) > 2]
    
    # Make a string from the tokens
    clean_tweets = ''.join(token+' ' for token in lemmatized_tokens)
    
    # Make a word cloud and display it
    word_cloud = WordCloud(max_font_size=50, max_words=100, background_color='white').generate(clean_tweets)
    
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
