import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


#tokenize of a line and returns an array of clean words
#clean words: without punctuations markers, stop words and lower case
#return array
def token_words(line):
     clean_words=[]
     tokens=nltk.word_tokenize(line)
     for word in tokens:
          if(word.isalpha()):
               clean_words.append(word.lower())
     return clean_words 

def build_frequency(data, number):
     dic_mf={}
     len=data.shape[0]
     for i in range(len):
          print(i, "of", len)
          text=data[i,0]
          clean_words=token_words(text)
          for word in clean_words:
             if(word in dic_mf):
                 dic_mf[word]+=1
             else: dic_mf[word]=1
               
     words_sorted=sorted(dic_mf.items(), key=lambda x: x[1], reverse=True)  
     freq_words={}
     
     wordsF=words_sorted[0:number]
     count=0
     for tup in wordsF:
          count+=1
          w,f=tup
          freq_words[w]=[count,f]
     
     return freq_words              

def build_frequency_classes(data,number,stop):
     dic_mf={}
     len=data.shape[0]
     for i in range(len):
          print(i, "of", len)
          text=data[i,0]
          clean_words=token_words(text)
          for word in clean_words:
             if(stop==True): 
                if not word in stop_words:       
                   if(word in dic_mf): dic_mf[word]+=1
                   else: dic_mf[word]=1
             else:
                 if(word in dic_mf): dic_mf[word]+=1
                 else: dic_mf[word]=1         
               
     words_sorted=sorted(dic_mf.items(), key=lambda x: x[1], reverse=True)  
     freq_words={}
     
     wordsF=words_sorted[0:number]
     count=0
     for tup in wordsF:
          count+=1
          w,f=tup
          freq_words[w]=[count,f]
     
     return freq_words      

#Dictionary:key is the label and value is array of the rows of that label 
#important to separate the data en classes
def table_labels(labels):
     dic_labels={}
     for i in range(len(labels)):
         key=labels[i]
         if(key in dic_labels):
              dic_labels[key].append(i)
         else:
             dic_labels[key]=[] 
                      
     return dic_labels 

#separate the data for each class. Return an array of matrix
#each component of the array is the matrix of a class
def organize_data(data,dic_labels):
     datas=[]
     for key in dic_labels:
          rows=dic_labels[key]
          datas.append(data[rows])
          
     return datas  


def write_table(table_f):
     outF = open("table_freq.txt", "w")
     words=[*table_f.keys()]
     arr=[*table_f.values()]
     size_bw=7
     
     for i in range(len(words)):
          word=words[i]
          word=word.ljust(15)
          rang=str(arr[i][0])
          rang=rang.ljust(10)
          line=word+rang+str(arr[i][1])
          outF.write(line)
          outF.write("\n")
     outF.close()
     
def write_table_classes(table_f,class_n):
     outF = open("table_freq_class_"+str(class_n)+".txt", "w")
     words=[*table_f.keys()]
     arr=[*table_f.values()]
     size_bw=7
     
     for i in range(len(words)):
          word=words[i]
          word=word.ljust(15)
          rang=str(arr[i][0])
          rang=rang.ljust(10)
          line=word+rang+str(arr[i][1])
          outF.write(line)
          outF.write("\n")
     outF.close()
 
def intersection_classes(table_classes):
     outF = open("intersection_words.txt", "w")
     words_classes=[]
     for i in range(len(table_classes)):
          table_f=table_classes[i]
          words=[*table_f.keys()]
          words_classes.append(words)
     
     words0=words_classes[0]
     words1=words_classes[1]
     words2=words_classes[2] 
     int1=np.intersect1d(words0,words1)
     int2=np.intersect1d(int1,words2)
    
     for word in int2:
          pos=[]
          for i in range(len(table_classes)):
              table_f=table_classes[i]  
              pos.append(table_f[word][0]) 
          
          line=word+"  "+str(pos[0])+"  "+str(pos[1])+"  "+str(pos[2])
          outF.write(line)
          outF.write("\n")
     outF.close()
     
     words_not_see0=[]
     words_not_see1=[]
     words_not_see2=[]
     
     for word in words0:
          if (word not in words1 and word not in words2): 
               words_not_see0.append(word)
     
     for word in words1:
          if (word not in words0 and word not in words2): 
               words_not_see1.append(word)          
                         
     for word in words2:
          if (word not in words0 and word not in words1): 
               words_not_see2.append(word) 
      
     outF2 = open("words_not_seen.txt", "w") 
     outF2.write("*** class0 ****"+"\n")         
     outF2.write("\n".join(str(item) for item in words_not_see0)) 
     outF2.write("\n"+"*** class1 ****"+"\n")         
     outF2.write("\n".join(str(item) for item in words_not_see1))
     outF2.write("\n"+"*** class2 ****"+"\n")
     outF2.write("\n".join(str(item) for item in words_not_see2))                             


data = pd.read_csv("train_posts.csv")
data=np.array(data)

#table_f=build_frequency(data, 1000)
#write_table(table_f)

data_train=data
label_train=data_train[:,-1]

dic_labels=table_labels(label_train)
data_org=organize_data(data_train,dic_labels) 


class2=data_org[1]
class0=data_org[0]
class1=data_org[2]
classes=[class0,class1,class2]

print(data.shape[0])
print(class0.shape[0])
print(class1.shape[0])
print(class2.shape[0])

table_classes=[]

for i in range(3):
      data=classes[i]
      table_f=build_frequency_classes(data,1000,True)  
      table_classes.append(table_f)
      write_table_classes(table_f,i)

intersection_classes(table_classes)








