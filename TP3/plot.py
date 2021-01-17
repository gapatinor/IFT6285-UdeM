import gensim.downloader as api
import pandas as pd
import numpy as np
import time
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import random

wv = api.load('word2vec-google-news-300')
print("finish loading")

def reduce_dimensions(words):
     num_dimensions = 2  # final num dimensions (2D, 3D, etc)

     vectors = [] # positions in vector space
     labels = [] # keep track of words to label our data again later
     for word in words:
         vectors.append(wv[word])
         labels.append(word)

     # convert both lists into numpy vectors for reduction
     vectors = np.asarray(vectors)
     labels = np.asarray(labels)

     # reduce using t-SNE
     tsne = TSNE(n_components=num_dimensions, random_state=0)
     vectors = tsne.fit_transform(vectors)

     x_vals = [v[0] for v in vectors]
     y_vals = [v[1] for v in vectors]
     return x_vals, y_vals, labels   
               


Corpus = pd.read_csv("voisins_radio.csv", names=['word','number_neighbors','closer_words'])[0:]
words=Corpus['word']
number_neigh=Corpus['number_neighbors']
closer_words=Corpus['closer_words']

x_vals=[]
y_vals=[]
labels_arr=[]
x_vals_arr=[]
y_vals_arr=[]
for ind in range(100):
     neighbours=closer_words[ind]
     neighbours_arr=neighbours.split(" ")
     word=words[ind]
     neighbours_arr.append(word)
     x_vals, y_vals, labels = reduce_dimensions(neighbours_arr)
     
     for label in labels:
         labels_arr.append(label) 
     for val in x_vals:
         x_vals_arr.append(val) 
     for val in y_vals:
         y_vals_arr.append(val) 
            
     plt.scatter(x_vals, y_vals)

indices = list(range(len(labels_arr)))
selected_indices = random.sample(indices, 15)
#selected_indices = indices
for j in selected_indices:
  plt.annotate(labels_arr[j], (x_vals_arr[j], y_vals_arr[j]))

plt.show()
#plt.savefig("neighs.pdf")
plt.close() 






