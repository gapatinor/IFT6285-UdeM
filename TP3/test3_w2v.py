import gensim.downloader as api
import pandas as pd
import numpy as np
import time
import csv
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def dic_most_similar(words,top,freq):
     tic = time.process_time()
     wv = api.load('word2vec-google-news-300')
     print("finish loading")
     
     count=1  
     similar_dic={}
     for word in words: 
        print(count) 
        if(word not in stop_words):
          try: 
             similar=wv.most_similar(positive=word, topn=top)
             sim_words=[]   
             for sim in similar:
               if(sim[1]>freq): sim_words.append(sim[0])
         
             similar_dic[word]=sim_words
          except:
             "not"
        #if(count==10): break
        count+=1
     
     toc = time.process_time()
     print("time:", toc-tic)   
        
     return similar_dic


#print the final file to submit to kaggle 
def write_results(final_similar):
     with open('voisins_radio.csv', mode='w') as csv_file:
         fieldnames = ['word','number neighbors','closer words']
         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
         writer.writeheader()
         for key in final_similar:
            closer_words=final_similar[key]
            number_neigh=len(closer_words)
            closer_words_write=closer_words[0:10]
            closer_words_write= " ".join(closer_words_write) 
            writer.writerow({'word': key, 'number neighbors':number_neigh, 'closer words': closer_words_write})

def similar_radio(most_similar, radi):
    final_similar={}
   
    for (key, value) in most_similar.items():
        tup_arr=value
        
        values=[]
        for word, freq in tup_arr:
             if(freq>radi and word not in stop_words): values.append(word)
             else: break
        final_similar[key]=(len(values),values)  
           
    return final_similar  


f=open('table_freq.txt', "r")
lines=f.readlines()

words=[]
for line in lines:
    line=line.split(" ")[0]
    words.append(line)

freq=0.4
similar_dic=dic_most_similar(words,500,freq)
write_results(similar_dic) 