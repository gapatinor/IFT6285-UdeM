import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

nrows_train=500000
nrows_test=10000

Corpus_test = pd.read_csv("normalise_test.csv", names=['text_final', 'class'])
Test_y_old=Corpus_test['class']

datafile = 'normalise.csv'
data_test= "normalise_test.csv"


#dont change these 2 lines
chunksize_train = 5000
chunksize_test = 100

count=1
predictions_svm_final=[]
predictions_xgboost_final=[]
predictions_adaboost_final=[]
for chunk_test in pd.read_csv(data_test, names=['text_final', 'class'], chunksize=chunksize_test, nrows=nrows_test):
  Test_X=chunk_test['text_final']
  Test_Y=chunk_test['class']
  predictions_svm=[]
  predictions_adaboost=[]
  for chunk in pd.read_csv(datafile, names=['text_final', 'class'], chunksize=chunksize_train, nrows=nrows_train):
     print(count)
    
     Train_X=chunk['text_final']
     Train_Y=chunk['class']
    
     Tfidf_vect = TfidfVectorizer()
     Tfidf_vect.fit(Train_X)
     Train_X_vec = Tfidf_vect.transform(Train_X)
     Test_X_vec = Tfidf_vect.transform(Test_X)
    
     
     model = svm.SVC(C=1, kernel='linear', gamma='auto')
     model.fit(Train_X_vec,Train_Y)
     prediction= model.predict(Test_X_vec)
     predictions_svm.append(prediction)
     
     print("doing adaboost")
     model3 = AdaBoostClassifier(n_estimators=100,learning_rate=1)
     model3.fit(Train_X_vec, Train_Y)
     prediction3= model3.predict(Test_X_vec)
     predictions_adaboost.append(prediction3)
     
     count+=1
    
  predictions_svm_final.append(np.rint(np.mean(predictions_svm,axis=0)))
  predictions_adaboost_final.append(np.rint(np.mean(predictions_adaboost,axis=0)))

pred=np.concatenate(predictions_svm_final)  
accuracy = accuracy_score(Test_y_old, pred)
print("Accuracy SVM: %.2f%%" % (accuracy * 100.0))
rep=classification_report(Test_y_old, pred)
print('report SVM %s' % rep)   
        
      
pred3=np.concatenate(predictions_adaboost_final)  
accuracy3 = accuracy_score(Test_y_old, pred3)
print("Accuracy adaboost: %.2f%%" % (accuracy3 * 100.0))  
rep=classification_report(Test_y_old, pred3)
print('report adaboost %s' % rep) 
  