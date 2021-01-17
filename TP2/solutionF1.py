import pandas as pd
import numpy as np
import nltk
import re, string, unicodedata
from collections import OrderedDict
import collections
import csv



def clean_text(text):
       
   replace={}       
   #split by several dots     
   sr = re.split("\.\.", text)

   replace["DOTS"]=0
   string_n=""
   for i in range(len(sr)):
       token=sr[i]
       if(i<len(sr)-1):
          string_n+=token+" DOTS"+" "
          replace["DOTS"]+=1
       else:
          string_n+=token      
   
   #split by several!
   sr = re.split("\!\!", string_n)
   
   replace["STRONG"]=0
   string_n=""
   for i in range(len(sr)):
      token=sr[i]
      if(i<len(sr)-1):
         string_n+=token+" STRONG"+" "
         replace["STRONG"]+=1
      else:
         string_n+=token
         
   #split by time format hour:min!
   pattern="\d:\d"
   sr = re.split(pattern, string_n)
   
   replace["TIME"]=0
   string_n=""
   for i in range(len(sr)):
      token=sr[i]
      if(i<len(sr)-1):
         string_n+=token+" TIME"+" "
         replace["TIME"]+=1
      else:
         string_n+=token
         
         
   #split by time format hour pm!
   pattern="\dpm"
   sr = re.split(pattern, string_n)
   
   string_n=""
   for i in range(len(sr)):
      token=sr[i]
      if(i<len(sr)-1):
         string_n+=token+" TIME"+" "
         replace["TIME"]+=1
      else:
         string_n+=token         
   
   #split by haha
   sr = re.split("haha", string_n)
   
   replace["RIRE"]=0
   string_n=""
   for i in range(len(sr)):
      token=sr[i]
      if(i<len(sr)-1):
         string_n+=token+" RIRE"+" "
         replace["RIRE"]+=1
      else:
         string_n+=token
   
   #split by hehe
   sr = re.split("hehe", string_n)
   
   string_n=""
   for i in range(len(sr)):
      token=sr[i]
      if(i<len(sr)-1):
         string_n+=token+" RIRE"+" "
         replace["RIRE"]+=1
      else:
         string_n+=token
   
   return (string_n, replace)       


def icons(text):
  replace={}     
     
  icons=[":D",":\)", ":-\)",':-]',':->','8-\)', ':\)',':-}', ':\)',':]', 
         ':3', ':>', '8\)', ':}', ':c\)',':^\)',':‑D']
  
  icons2= [":‑\(",":\(", ":-c",":-<",":<",":‑\[",":\[", " >:\[",  ":{"]	
  
  replace["SMILY"]=0      
  string_n=text
  for icon in icons:     
     sr = re.split(icon, string_n)
     #icon found
     if(len(sr)>1):
         string_n=""
         for i in range(len(sr)):
            token=sr[i]
            if(i<len(sr)-1):
               string_n+=token+" SMILY"+" "
               replace["SMILY"]+=1
            else:
               string_n+=token  
  
  replace["SAD"]=0
  for icon in icons2:     
     sr = re.split(icon, string_n)
     #icon found
     if(len(sr)>1):
         string_n=""
         for i in range(len(sr)):
            token=sr[i]
            if(i<len(sr)-1):
               string_n+=token+" SAD"+" "
               replace["SAD"]+=1
            else:
               string_n+=token
     
  return (string_n,replace)
                

def final_text(text):
    ft=""
    tokens=nltk.word_tokenize(text) 
    suspect=[] 
        
    for token in tokens:
         results = collections.Counter(token)  
         max_value = max(results.values())
         if(max_value>=3):
            suspect.append(token)  
              
         result=token 
         if(result.isalpha()):  
            if(result=="STRONG"): result="STRONG!"
            ft+=result+" "       
            
         
    return ft, suspect 

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


def write_output(data_train):
     with open('normalise.csv', mode='w') as csv_file:
         fieldnames = ['Text', 'Category']
         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
         
         #writer.writeheader()
         for i in range(data_train.shape[0]):
            text=data_train[i][0]
            cat=data_train[i][1]
            
            string_1,replace1=clean_text(text)
            string_2,replace2=icons(string_1)
            string_f,suspect=final_text(string_2)  
            text_f="\"" + string_f + "\""
            writer.writerow({'Text': text_f, 'Category': cat})


def classes_norm(data_train, class_n):
     replace_f={}
     suspect_f=[]
     for i in range(data_train.shape[0]):
        text=data_train[i][0]
        cat=data_train[i][1]
        
        string_1,replace1=clean_text(text)
        for key1 in replace1:
             value=replace1[key1]
             if(key1 in replace_f): replace_f[key1]+=value
             else: replace_f[key1]=value
             
        string_2,replace2=icons(string_1)
        for key2 in replace2:
             value=replace2[key2]
             if(key2 in replace_f): replace_f[key2]+=value
             else: replace_f[key2]=value
        
        string_f,suspect=final_text(string_2)
        for word in suspect:
             suspect_f.append(suspect) 
     
                
        #text_f="\"" + string_f + "\""
        #writer.writerow({'Text': text_f, 'Category': cat})
     return (replace_f,suspect)


df=pd.read_csv('train.csv', sep=',',header=None)
data=df.values
data_train=data
label_train=data[:,-1]   
 
dic_labels=table_labels(label_train)
data_org=organize_data(data_train,dic_labels) 

'''
class0=data_org[1]
class1=data_org[0]
class2=data_org[2]

classes=[class0,class1,class2]
replace_class=[]
suspect_class=[]

for i in range(3):
      data_train=classes[i]
      print("size of class",i," :", data_train.shape[0])
      replace,suspect=classes_norm(data_train, i)
      replace_class.append(replace)
      suspect_class.append(suspect)
 
print(replace_class)
print(suspect_class)'''      


                
#df=pd.read_csv('train.csv', sep=',',header=None)
#data=df.values
#data_train=data
write_output(data_train)



 
   
