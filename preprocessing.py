import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import pickle

def tweet_to_words(review, remove_stopwords=True):

    tweet_text = re.sub("https?:\/\/.*[\r\n]*\/", '', review)
    tweet_text = BeautifulSoup(tweet_text,"html.parser").get_text()
    tweet_text = re.sub("[1-9;:,\'\"@!~?%$^&`./\<>*\-+)(]", " ", tweet_text)


    words = tweet_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        words = [w for w in words if w[0]!="@"]

    return (words)



# files reading
train = pd.read_csv("data/training.csv", header=0,delimiter=",",encoding="ISO-8859-1");

print(train["text"][1])

tweets = []

for tweet in train["text"][0:500000]:
    tweets.append(tweet_to_words(tweet))


tweet_with_hashtag=[]
for tweet in tweets:
    add = False
    hashtags = []
    for word in tweet:
        if word[0]=="#" and len(word)>1:
            add=True
            hashtags.append(word)
            break
    if add:
        tweet_with_hashtag.append((tweet,hashtags))

print(len(tweet_with_hashtag))
print(tweet_with_hashtag[10])

with open('Tweets_with_hashtag.500tweet', 'wb') as fp:
    pickle.dump(tweet_with_hashtag, fp)