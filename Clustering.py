from gensim.models import Doc2Vec,doc2vec
import pickle
import numpy
from sklearn.cluster import DBSCAN,KMeans
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)


model = Doc2Vec.load("model.d2v")

with open('Tweets_with_hashtag.500tweet', 'rb') as pickle_file:
    # pickle_file.read()
    tweet_with_hashtag = pickle.load(pickle_file)

test_arrays = numpy.zeros((len(tweet_with_hashtag), 300))

for i in range(0,len(tweet_with_hashtag)):
    test_arrays[i] = (model.infer_vector(tweet_with_hashtag[i][0])*2)-1

print(len(test_arrays))
print(len(tweet_with_hashtag))
db = DBSCAN(eps=3, min_samples=2, metric='manhattan').fit(test_arrays)

labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


print(n_clusters_)
# print(model.most_similar("obama"))




with open('model1.7.2.dbscan', 'wb') as fp:
    pickle.dump(db, fp)



