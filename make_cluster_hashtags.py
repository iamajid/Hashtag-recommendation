import pickle
from gensim.models import Doc2Vec
from sklearn.cluster import DBSCAN
import numpy
import time,random



with open('Tweets_with_hashtag.tweet', 'rb') as pickle_file:
    # pickle_file.read()
    tweet_with_hashtag = pickle.load(pickle_file)

with open('modelK1', 'rb') as pickle_file:
    # pickle_file.read()
    db = pickle.load(pickle_file)

model = Doc2Vec.load("model.d2v")



test_arrays = numpy.zeros((len(tweet_with_hashtag), 300))

for i in range(0,int(len(tweet_with_hashtag)*0.8)):
    test_arrays[i] = model.infer_vector(tweet_with_hashtag[i][0])

labels = db.labels_

# db.labels_ = set(labels)
print(db.labels_)

# labels = db.labels_

# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)




# print(clusters)
all_hashtags = {}
sorted_={}
for i in range(800):
    all_hashtags[i] = {}
    sorted_[i]={}

print(len(all_hashtags))


a = db.fit_predict(test_arrays)


i=0
print(max(list(a)))

p = [i for i in list(a) if i!=-1]

for cl in list(p):

    for hashtag in tweet_with_hashtag[i][1]:
        print(hashtag)
        if hashtag in all_hashtags[cl]:
            all_hashtags[cl][hashtag]+=1
        else:
            all_hashtags[cl][hashtag]=1
    i+=1




import operator



i=0

for cl_hash in all_hashtags.values():
    sorted_d = sorted(cl_hash.items(), key=operator.itemgetter(1),reverse=True)
    sorted_[i]=sorted_d
    i+=1

del all_hashtags

file = open("hash.txt","w+")
for l in sorted_.values():
    file.write(str(l))
    file.write("\n")

with open("hashtag_clusters_kmean.pkl","wb") as f:
    pickle.dump(sorted_,f)

# print(len(all_hashtags))

