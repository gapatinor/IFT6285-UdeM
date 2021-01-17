from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import pandas as pd
import csv
     

def morpho2(word, neighbor):
     lemmatizer=WordNetLemmatizer()
     
     word=word.lower()
     word1 = lemmatizer.lemmatize(neighbor.lower(), pos = "n")
     word2 = lemmatizer.lemmatize(word1, pos = "v")
     word3 = lemmatizer.lemmatize(word2, pos = ("a"))
     if(word==word3): 
             str=neighbor+"["+"MORPHO"+"]"
             return (True,str)
     else: return (False,word)        

def morpho(word, neighbor):
     neighbor_word=neighbor
     try:
       word = wn.synsets(word)[0]
       neighbor = wn.synsets(neighbor)[0]
       if(word==neighbor):
            str=neighbor_word+"["+"MORPHO"+"]"
            return (True,str)
       else:
            return (False,neighbor_word)
     except:
            return (False,neighbor_word)     

def hyper(word, neighbor):
     neighbor_word=neighbor
     try:
       word = wn.synsets(word)[0]
       neighbor = wn.synsets(neighbor)[0]
     
       hyper_word = set([i for i in word.closure(lambda s:s.hypernyms())])
       if(neighbor in hyper_word):
          str=neighbor_word+"["+"HYPER"+"]"
          return (True,str)
       else:
          return (False,neighbor_word)
     except:
          return (False,neighbor_word)      
               
def hypo(word, neighbor):
     neighbor_word=neighbor
     try:
       word = wn.synsets(word)[0]
       neighbor = wn.synsets(neighbor)[0]
    
       hypo_word = set([i for i in word.closure(lambda s:s.hyponyms())])
       if(neighbor in hypo_word):
          str=neighbor_word+"["+"HYPO"+"]"
          return (True,str)
       else:
          return (False,neighbor_word)               
     except:
          return (False,neighbor_word)
          

def cohypo(word,neighbor):
     neighbor_word=neighbor
     try:
       word = wn.synsets(word)[0]
       neighbor = wn.synsets(neighbor)[0]
    
       hyper_word = set([i for i in word.closure(lambda s:s.hypernyms())])
       hyper_neigh = set([i for i in neighbor.closure(lambda s:s.hypernyms())])
  
       intersection=hyper_word.intersection(hyper_neigh)
       if(len(intersection)!=0):
           str=neighbor_word+"["+"COHYPO"+"]"
           return (True,str)
       else:
        return (False,neighbor_word)   
     except:
          return (False,neighbor_word)        

def synonyms(word, neighbor):
     synonyms=[]
     try:
       for syn in wn.synsets(word):
       		for l in syn.lemmas():
       			synonyms.append(l.name())
                     
       if(neighbor in synonyms):
            str=neighbor+"["+"SYN"+"]"
            return (True,str)		
       else:
        return (False,neighbor)
     except:
        return (False,neighbor)

def antonyms(word, neighbor):
     antonyms=[]
     try:
       for syn in wn.synsets(word):
       		for l in syn.lemmas():
                    if l.antonyms(): antonyms.append(l.antonyms()[0].name())
                     
       if(neighbor in antonyms):
            str=neighbor+"["+"ANTO"+"]"
            return (True,str)		
       else:
        return (False,neighbor)
     except:
        return (False,neighbor)

def holonym(word,neighbor):
     neighbor_word=neighbor
     holonym=[]
     try:
       word = wn.synsets(word)[0]     
       for syn in wn.synsets(neighbor):
          for l in syn.part_holonyms():
             holonym.append(l) 
       if(word in holonym):
           str=neighbor_word+"["+"PARTOF"+"]"
           return (True,str)
       else:
           return (False,neighbor_word)                    
     except:
          return (False,neighbor_word)
               
def build_labels(word, neighbors):
     number_mods=0
     number_cohypos=0
     for i in range(len(neighbors)):
          label=False
          neigh=neighbors[i]
          
          fl,word_l=morpho2(word,neigh)
          if(fl==True): 
               neighbors[i]=word_l
               label=True
               number_mods+=1
               
          fl,word_l=hypo(word,neigh)
          if(label==False and fl==True):
               neighbors[i]=word_l
               label=True
               number_mods+=1
          
          fl,word_l=synonyms(word,neigh)
          if(label==False and fl==True):
               neighbors[i]=word_l
               label=True
               number_mods+=1
               
          fl,word_l=antonyms(word,neigh)
          if(label==False and fl==True):
               neighbors[i]=word_l
               label=True  
               number_mods+=1        
          
          fl,word_l=holonym(word,neigh)
          if(label==False and fl==True):
               neighbors[i]=word_l
               label=True
               number_mods+=1
               
          fl,word_l=cohypo(word,neigh)
          if(label==False and fl==True):
               neighbors[i]=word_l
               label=True   
               number_mods+=1  
               number_cohypos+=1
     return (neighbors, number_mods-number_cohypos)               

#print the final file to submit to kaggle 
def write_results(final_similar):
     with open('voisins_labels.csv', mode='w') as csv_file:
         fieldnames = ['word','number neighbors','label words']
         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
         writer.writeheader()
         for element in final_similar:
            key=element[0]
            tup=element[1] 
            number_mods=tup[0]
            number_neigh=tup[1] 
            labels=tup[2] 
           
            labels_write= " ".join(labels) 
            writer.writerow({'word': key, 'number neighbors':number_neigh, 'label words': labels_write})


Corpus = pd.read_csv("voisins_radio.csv", names=['word','number_neighbors','closer_words'])[0:]
words=Corpus['word']
number_neigh=Corpus['number_neighbors']
closer_words=Corpus['closer_words']

dic_labels={}
for i in range(len(words)):
     word=words[i]
     neighbors=closer_words[i].split(" ")
     labels, number_mods=build_labels(word, neighbors)
     dic_labels[word]=(number_mods,number_neigh[i],labels)


dic_sorted=sorted(dic_labels.items(), key=lambda x: x[1], reverse=True)
write_results(dic_sorted[0:500])

