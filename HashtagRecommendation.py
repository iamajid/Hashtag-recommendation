import pickle
import numpy
from gensim.models import Doc2Vec



with open('hashtag_clusters_kmean.pkl', 'rb') as pickle_file:
    # pickle_file.read()
    all_hashtags = pickle.load(pickle_file)



with open('Tweets_with_hashtag.tweet', 'rb') as pickle_file:
    # pickle_file.read()
    tweet_with_hashtag = pickle.load(pickle_file)

with open('modelK1', 'rb') as pickle_file:
    # pickle_file.read()
    db = pickle.load(pickle_file)

model = Doc2Vec.load("model.d2v")

test_arrays = numpy.zeros((int(0.2*len(tweet_with_hashtag)), 300))
for i in range(0,int(0.2*len(tweet_with_hashtag))):
    test_arrays[i] = model.infer_vector(tweet_with_hashtag[i+int(0.8*len(tweet_with_hashtag))][0])


a = db.fit_predict(test_arrays)

recommended = []
print(len(list(a)))
for i in range(len(list(a))):
    print(a[i])
    if a[i]==-1:
        continue
    if len(all_hashtags[list(a)[i]])>5:
        RH =all_hashtags[list(a)[i]][0:4]
    else:
        RH = all_hashtags[list(a)[i]]
    recommended.append((tweet_with_hashtag[i+int(0.8*len(tweet_with_hashtag))],RH))


file = open("RH.txt","w+")
for l in recommended:
    file.write(str(l))
    file.write("\n")

