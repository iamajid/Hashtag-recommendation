
from gensim.models import Doc2Vec,doc2vec
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

import time,random





start = time.time()

def tweet_to_words(review, remove_stopwords=True):

    tweet_text = re.sub("https?:\/\/.*[\r\n]*\/", '', review)
    tweet_text = BeautifulSoup(tweet_text,"html.parser").get_text()
    tweet_text = re.sub("[1-9;:,!~?%$^&`./\<>*\-+)(]", " ", tweet_text)


    words = tweet_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        words = [w for w in words if w[0]!="@"]

    return (words)



# files reading
train = pd.read_csv("data/training.csv", header=0,delimiter=",",encoding="ISO-8859-1");

print(train["text"].shape)

tweets = []

for tweet in train["text"][0:500000]:
    tweets.append(tweet_to_words(tweet))


i=0
readyd=[]
for sen in tweets:
    ss = doc2vec.TaggedDocument(words = sen , tags = [str(i)])
    i=i+1
    readyd.append(ss)




# readyd = doc2vec.TaggedDocument()



import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

size= 300        # Word vector dimensionality
min_count = 1    # Minimum word count
workers = 8      # Number of threads to run in parallel
window = 300     # Context window size


model = doc2vec.Doc2Vec(readyd,size = size, window = window, min_count = min_count, workers=workers)

for epoch in range(3):
    shuffled = list(readyd)
    random.shuffle(shuffled)
    model.train(shuffled,total_examples=model.corpus_count,
                total_words=None, epochs=1, start_alpha=None, end_alpha=None, word_count=0,
                queue_factor=2, report_delay=1.0)
    print ("///////////////////")

model.save("model.d2v")



end = time.time()
elapsed = end - start
print ("\nTime taken for modeling : ", int(elapsed/3600), "houres and" ,abs( int((elapsed-int(elapsed/3600)*3600)/60)),
       " minutes and " , \
    abs(int(elapsed-int(elapsed/3600)*3600 -  int((elapsed-int(elapsed/3600)*3600)/60)*60)) , " seconds")
