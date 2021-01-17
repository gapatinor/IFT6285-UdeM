import csv
import pandas as pd
import numpy as np

from gensim.models.doc2vec import Doc2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 

#gensim requires the data set to be labeled
def label_sentences(corpus, label_type):
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v, [label]))
        
    return labeled
    
def get_vectors(model, corpus_size, vectors_size, vectors_type):
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors    
 
#reading the data set    
Corpus_train = pd.read_csv("normalise.csv", names=['text_final', 'class'])
Corpus_test = pd.read_csv("normalise_test.csv", names=['text_final', 'class'])

X_train=Corpus_train['text_final']
y_train=Corpus_train['class']
X_test=Corpus_test['text_final']
y_test=Corpus_test['class']
 
#label the data set    
X_train=label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')
all_data = X_train + X_test
 
#build the model with doc2vec    
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab(all_data)
 
#converge the model  
for epoch in range(30):
    print('iteration {0}'.format(epoch))
    model_dbow.train(all_data,
                total_examples=model_dbow.corpus_count,
                epochs=1)
    # decrease the learning rate
    model_dbow.alpha -= 0.0002
    # fix the learning rate, no decay
    model_dbow.min_alpha = model_dbow.alpha

#build the vector features (transformation of text in vectors)
train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')

#build the SVM linear classifier
text_clf = SVC(C=1,gamma='auto')
#fit the vectors 
text_clf.fit(train_vectors_dbow, y_train)
#prediction
y_pred = text_clf.predict(test_vectors_dbow)
#build report 
rep=classification_report(y_test, y_pred)
print('accuracy SVM %s' % accuracy_score(y_pred, y_test))
print('report SVM %s' % rep)


 
   
   
   