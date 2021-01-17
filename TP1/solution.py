import numpy as np
import matplotlib.pyplot as plt
import nltk 
import time


#build an array of tokens from the text
def build_tokens(f):
    tokens = nltk.word_tokenize(f.read())
    f.close()
    return tokens


# given a list of tokens, return a dictionary where key are the existing unique tokens
# and values are their counts. 
def build_dic_tokens(tokens):
    unigram_table = {}
    for token in tokens:
        if token in unigram_table:
            unigram_table[token] += 1
        else:
            unigram_table[token] = 1
    return unigram_table

def count_words(dic_tokens_train, dic_tokens_test):
    words_100=0
    for key in dic_tokens_train:
       if(dic_tokens_train[key]>=100):
           words_100+=1
    
    words_non_vu=0
    for key in dic_tokens_test:  
        if(key not in dic_tokens_train):
           words_non_vu+=1
           
    return (words_100, words_non_vu)       

def bigram(tokens):
    unigrams=build_dic_tokens(tokens)
    length=len(unigrams)
    bigram_table={}
    num_bigrams=0
    
    for i in range(len(tokens)-1):
        if tokens[i] in bigram_table:
            if(tokens[i+1] in bigram_table[tokens[i]]):
               bigram_table[tokens[i]][tokens[i+1]] += 1
            else:
               bigram_table[tokens[i]][tokens[i+1]] = 1  
               num_bigrams+=1
               
        else:
           bigram_table[tokens[i]]={} 
           bigram_table[tokens[i]][tokens[i+1]] = 1   
           num_bigrams+=1
    
    return (bigram_table, unigrams, num_bigrams)        
     
def get_probability(prefix, suffix, unigram_dic, bigram_dic):
    count = 0
    if suffix in bigram_dic[prefix]:
        count = bigram_dic[prefix][suffix]
    if count < 5:
        count = 1
            
    return (round(count / (unigram_dic[prefix]),5))                        

def build_unknown(tokens, limit):
    unigrams=build_dic_tokens(tokens)
    i=0
    for token in tokens:
        if(unigrams[token]<limit):
            tokens[i]="UNK"
        i+=1
        
    return tokens      

def perplexity(test_token):
    length = len(test_token)
    for i in range(length):
        if test_token[i] not in unigram_unk:
            test_token[i] = 'UNK'
    
    sum = 0.0
    for i in range(length - 1):
        prob = get_probability(test_token[i], test_token[i+1], unigram_unk, bigram_unk)
        sum+=((np.log(prob))*(-1))

    return np.exp((sum / length))

def random_limits_generator(tokens_test):
    
    while(True):
      limits=np.random.randint(0, len(tokens_test),2)
      if(np.abs(limits[1]-limits[0])>=7000): break
    return(np.sort(limits))

def random_test(tokens_test):
    perp=[]
    for i in range(2):
       print("step: ",i)    
       limits=random_limits_generator(tokens_test)
       tokens_random=tokens_test[limits[0]:limits[1]]
       p=perplexity(tokens_random)
       perp.append(p)
    return (np.mean(perp), np.std(perp))   
  
def prediction_words(tokens_test):
    symbols=[".",",","?","the","to","-",";"]
    L=[]
    while(True):
      rn=np.random.randint(0, len(tokens_test))
      random_word_test=tokens_test[rn]
      if(random_word_test not in symbols):
          break
      
    neighbors=bigram_unk[random_word_test]
    table_prob={}
    sorted_table={}
    
    for neigh in neighbors:
         prob=get_probability(random_word_test, neigh, unigram_unk, bigram_unk)  
         if(prob>=1e-5): 
           table_prob[neigh]=prob
           
    for key, value in sorted(table_prob.items(), key=lambda item: item[1]):
         sorted_table[key]=value  
         t=(key,value)
         L.append(t)   
            
    return random_word_test, L[len(L):len(L)-10:-1]
    
def bigrams_test_prob(tokens_test,random_word_test):
     bigram_test, unigram_test, num_bigram_test=bigram(tokens_test)
     neighbors= bigram_test[random_word_test]
     table_prob={}
     L=[]
     
     for neigh in neighbors:
          prob=get_probability(random_word_test, neigh, unigram_test, bigram_test)  
          if(prob>=1e-4): 
            table_prob[neigh]=prob
            
     for key, value in sorted(table_prob.items(), key=lambda item: item[1]):
          t=(key,value)
          L.append(t)
          
     return random_word_test, L[len(L):len(L)-10:-1]      
  
def unigram_test_prob(test_token):
    symbols=[".",",","?","the","to","-",";"]
    length=len(tokens_test)
    for i in range(length):
        if test_token[i] not in unigram_unk:
            test_token[i] = 'UNK'
    
    max_prob={}        
    for word in test_token:
         if(word not in symbols):
           prob=(unigram_unk[word])/len(tokens_unk)
           max_prob[word]=prob
         
    L=[]
    for key, value in sorted(max_prob.items(), key=lambda item: item[1]):
         t=(key,value)
         L.append(t)
    
    return L[len(L):len(L)-10:-1]       
  
def most_commun_test(tokens_test):
    bigram_test, unigram_test, num_bigram_test=bigram(tokens_test)
    symbols=[".",",","?","the","to","-",";"]
    L=[]
    for key, value in sorted(unigram_test.items(), key=lambda item: item[1]):
         if(key not in symbols):
          t=(key)
          L.append(t)
         
    return L[len(L):len(L)-10:-1]
      
    
t1=time.process_time()
f_train = open("tp1_valid/valid.en")
tokens_train=build_tokens(f_train)
tokens_unk=build_unknown(tokens_train, limit=30)
bigram_unk, unigram_unk, num_bigram_unk=bigram(tokens_unk)
t2=time.process_time()
print("process time with UNK: ",t2-t1,"\n")

f_test = open("tp1_test/test.en")
tokens_test=build_tokens(f_test)
perp=perplexity(tokens_test)
print("perplexity in the bigram model: ",perp,"\n")

perp_f=random_test(tokens_test)
print("mean and std of random phrases in test: ", perp_f,"\n")

word,table_prob=prediction_words(tokens_test)
print("random word in test and its more common suffix words:")
print(word, "-->", table_prob,"\n")

print("random word in test and its more common suffix words in the bigram of test:")
word, table_prob=bigrams_test_prob(tokens_test, word)
print(word, "-->", table_prob,"\n")

max_prob=unigram_test_prob(tokens_test)
print("more probably tokens in test: ", max_prob,"\n")

most_c=most_commun_test(tokens_test)
print("more probably tokens in test in the unigram of test: ", most_c)

