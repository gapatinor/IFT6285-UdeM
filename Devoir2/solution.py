import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize

stops=stopwords.words('english')

def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.savefig("features_classes"+str(df.label)+".pdf")

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    
    return top_tfidf_feats(tfidf_means, features, top_n)


def plot_features_TFIDF(X_train,Y_train):
      tfidf = TfidfVectorizer(max_features = 500, ngram_range = (1,1),stop_words=stops)
      X_train_tfidf=tfidf.fit_transform(X_train)
     
      features=tfidf.get_feature_names()
      top_mean_feats(X_train_tfidf,features)
      dfs=top_feats_by_class(X_train_tfidf, Y_train, features, min_tfidf=0.1, top_n=25)
      plot_tfidf_classfeats_h(dfs)

def identify_adjectives(X_train, label):
     adjective_tags = ["JJ", "JJR", "JJS"]
     adjectives_list=[]
     
     for text in X_train:
          tokens=word_tokenize(text)
          tags=nltk.pos_tag(tokens)
          for tag in tags:
               if(label==True):
                 if(tag[1] in adjective_tags):
                    if(tag[0]!="good" and tag[0]!="great"): adjectives_list.append(tag[0])
               else:
                  if(tag[1] in adjective_tags):
                     adjectives_list.append(tag[0])    
          
     return adjectives_list      
 
def build_big_text(big_list):
     text_f=" "
     for text in big_list:
          text_f=text_f+text+" "
     return text_f

def plot_world_text(text_f, name):
     wc = WordCloud(max_words= 150,
                    width = 800, 
                    height = 800,
                    background_color ='white',
                    contour_width=3, 
                    contour_color='steelblue',
                    stopwords=stops,
                    min_font_size = 1).generate(text_f) 

     plt.imshow(wc) 
     plt.axis("off")  
     plt.savefig(name)
     plt.close()
               
data_pos = pd.read_csv('comments_pos.csv')
X_train_pos=data_pos['Comments']
Y_train_pos=np.ones(len(X_train_pos))

data_neg = pd.read_csv('comments_neg.csv')
X_train_neg=data_neg['Comments']
Y_train_neg=np.zeros(len(X_train_neg))

#plot_features_TFIDF(X_train_pos,Y_train_pos)
#plot_features_TFIDF(X_train_neg,Y_train_neg)

pos_adjectives=identify_adjectives(X_train_pos, label=False)          
text_pos_adjectives=build_big_text(pos_adjectives)
plot_world_text(text_pos_adjectives, "adjectives_pos.pdf")

neg_adjectives=identify_adjectives(X_train_neg, label=True)          
text_neg_adjectives=build_big_text(neg_adjectives)
plot_world_text(text_neg_adjectives, "adjectives_neg.pdf")

text_pos=build_big_text(X_train_pos)
plot_world_text(text_pos, "words_pos.pdf")

text_neg=build_big_text(X_train_neg)
plot_world_text(text_neg, "words_neg.pdf")

'''
text_f_pos=" "
for text in pos_adjectives:
     text_f_pos=text_f_pos+text+" "
     
wc = WordCloud(max_words= 100,
                      width = 800, 
                      height = 800,
                      background_color ='white',
                      contour_width=3, 
                      contour_color='steelblue',
                      stopwords=stops,
                      min_font_size = 1).generate(text_f_pos) 

#plt.figure(figsize = (14, 14)) 
plt.imshow(wc) 
plt.axis("off")  
plt.savefig("adjective_pos2.pdf")
plt.close()  '''
