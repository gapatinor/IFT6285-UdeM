import csv
import pandas as pd
import re, string

from nltk import pos_tag
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier


def norm_text(text):
   #split by several dots     
   sr = re.split("\.\.", text)

   
   string_n=""
   for i in range(len(sr)):
       token=sr[i]
       if(i<len(sr)-1):
          string_n+=token+" DOTS"+" "
          
       else:
          string_n+=token      
   
   #split by several!
   sr = re.split("\!\!", string_n)
   
   
   string_n=""
   for i in range(len(sr)):
      token=sr[i]
      if(i<len(sr)-1):
         string_n+=token+" STRONG"+" "
         
      else:
         string_n+=token
         
   #split by time format hour:min!
   pattern="\d:\d"
   sr = re.split(pattern, string_n)
   
   
   string_n=""
   for i in range(len(sr)):
      token=sr[i]
      if(i<len(sr)-1):
         string_n+=token+" TIME"+" "
         
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
         
      else:
         string_n+=token         
   
   #split by haha
   sr = re.split("haha", string_n)
   
  
   string_n=""
   for i in range(len(sr)):
      token=sr[i]
      if(i<len(sr)-1):
         string_n+=token+" RIRE"+" "
         
      else:
         string_n+=token
   
   #split by hehe
   sr = re.split("hehe", string_n)
   
   string_n=""
   for i in range(len(sr)):
      token=sr[i]
      if(i<len(sr)-1):
         string_n+=token+" RIRE"+" "
         
      else:
         string_n+=token
   
   return string_n


def clean_text(Corpus):
   with open('normalise.csv', mode='w') as csv_file:
     fieldnames = ['text_final', 'class']
     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
     print("remove blanck")
     Corpus['blog'].dropna(inplace=True)
     
     print("lower case")
     Corpus['blog'] = [entry.lower() for entry in Corpus['blog']]
    
    
     # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
     # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
     print("lemma")
     tag_map = defaultdict(lambda : wn.NOUN)
     tag_map['J'] = wn.ADJ
     tag_map['V'] = wn.VERB
     tag_map['R'] = wn.ADV
    
     for index,entry in enumerate(Corpus['blog']):
        print("index", index, "of ", Corpus.shape[0])
        
        #normalizing
        entry=norm_text(entry)
        
        tokens=word_tokenize(entry)
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(tokens):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final=word
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index,'text_final'] = str(Final_words)
        writer.writerow({'text_final': str(Final_words), 'class': Corpus['class'][index]})
        
   
Corpus_train = pd.read_csv("examples/train_posts.csv", names=['blog', 'class']).iloc[0:250000]
Corpus_train_clean=clean_text(Corpus_train)







