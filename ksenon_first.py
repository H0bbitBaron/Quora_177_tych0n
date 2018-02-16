#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:42:39 2017

@author: pavel
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn import linear_model
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix
import sys
import os
# reload(sys)
# sys.setdefaultencoding('utf8')
import string
import random
import math
###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import seaborn as sns

from wordcloud import WordCloud

import hyperopt
# from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import nltk
from nltk.corpus import stopwords
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


import xgboost as xgb

pal = sns.color_palette()

train_csv = "/home/pavel/anaconda3/Scripts/Quora/train.csv"
test_csv = "/home/pavel/anaconda3/Scripts/Quora/test.csv"


train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)



train_data.info()

train_data.head(10)

# Text Analysis #

train_qs = pd.Series(train_data['question1'].tolist() + train_data['question2'].tolist()).astype(str)
test_qs = pd.Series(test_data['question1'].tolist() + test_data['question2'].tolist()).astype(str)


### EXPLORATORY ANALYSIS ###

# dist_train = train_qs.apply(len) # Character quantity in each train question
# dist_test = test_qs.apply(len) # Character quantity in each test question

# plt.figure(figsize = (15,10))
# plt.hist(dist_train, bins = 200, range = [0,200], color = pal[2], normed = True, label = 'Train')
# plt.hist(dist_test, bins = 200, range = [0,200], color = pal[1], normed = True, alpha = 0.5,label = 'Test')
# plt.title('Normalized histogram of character quantity in questions', fontsize = 16)
# plt.legend()
# plt.xlabel('Character Quantity', fontsize = 16)
# plt.ylabel('Probability', fontsize = 16)

# Statistics
# print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'
#      .format(dist_train.mean(), dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))

## Do the same for word quantity ##

# dist_train = train_qs.apply(lambda x: len(x.split(' ')))
# dist_test = test_qs.apply(lambda x: len(x.split(' ')))

# plt.figure(figsize=(15, 10))
# plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True, label='train')
# plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], normed=True, alpha=0.5, label='test')
# plt.title('Normalised histogram of word count in questions', fontsize=15)
# plt.legend()
# plt.xlabel('Number of words', fontsize=15)
# plt.ylabel('Probability', fontsize=15)

# print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
#                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))


### Let's see WordCloud Diagram based on our questions ###

 cloud = WordCloud(width = 1440, height = 1080).generate(" ".join(train_qs.astype(str)))
 plt.figure(figsize = (20,15))
 plt.imshow(cloud)
 plt.axis('off')


#########################################

# Semantic Analysis #

qmarks = np.mean(train_qs.apply(lambda x: '?' in x))
math = np.mean(train_qs.apply(lambda x:'[math]' in x))
fullstop = np.mean(train_qs.apply(lambda x: '.' in x))
capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))

capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))

Trump = np.mean(train_qs.apply(lambda x:'Donald Trump ' in x))
Diff = np.mean(train_qs.apply(lambda x:' difference ' in x))
BW = np.mean(train_qs.apply(lambda x:'best way' in x))


print('Questions with question marks: {:.2f}%'.format(qmarks * 100))
print('Questions with [math] tags: {:.2f}%'.format(math * 100))
print('Questions with full stops: {:.2f}%'.format(fullstop * 100))
print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))
print('Questions with capital letters: {:.2f}%'.format(capitals * 100))
print('Questions with numbers: {:.2f}%'.format(numbers * 100))

set = pd.DataFrame()
del set

set['how'] = train_data['question1'].apply(lambda x: 'How much' in x)
set.how = set.how.astype(int)

########################################

### Feature Engineering ###

# WORD_MATCH_SHARE' Feature 
stops = set(stopwords.words("english"))  



def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


# plt.figure(figsize=(15, 5))

train_word_match = train_data.apply(word_match_share, axis=1, raw=True)

# plt.hist(train_word_match[train_data['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
# plt.hist(train_word_match[train_data['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
# plt.legend()
# plt.title('Label distribution over word_match_share', fontsize=15)
# plt.xlabel('word_match_share', fontsize=15)


#Now try TF-IDF with weights 

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


# print('Most common words and weights: \n')
# print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
# print('\nLeast common words and weights: ')
# (sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])



def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

# plt.figure(figsize=(15, 5))

tfidf_train_word_match = train_data.apply(tfidf_word_match_share, axis=1, raw=True)

# plt.hist(tfidf_train_word_match[train_data['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')
# plt.hist(tfidf_train_word_match[train_data['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')
# plt.legend()
# plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
# plt.xlabel('word_match_share', fontsize=15)

## Prepare data for modelling

X_train = pd.DataFrame()
X_test = pd.DataFrame()

X_train['word_match'] = train_word_match
X_test['word_match'] = test_data.apply(word_match_share, axis = 1, raw = True)

X_train['tfidf_word_match'] = tfidf_train_word_match
X_test['tfidf_word_match'] = test_data.apply(tfidf_word_match_share, axis = 1, raw = True)

Y_train = train_data['is_duplicate'].values


train_que1 = pd.Series(train_data['question1'].tolist()).astype(str)
train_que2 = pd.Series(train_data['question2'].tolist()).astype(str)

test_que1 = pd.Series(test_data['question1'].tolist()).astype(str)
test_que2 = pd.Series(test_data['question2'].tolist()).astype(str)

# Amount of characters in each question
X_train['num_char_q1'] = train_que1.apply(lambda x: len(x))
X_train['num_char_q2'] = train_que2.apply(lambda x: len(x))

X_test['num_char_q1'] = test_que1.apply(lambda x: len(x))
X_test['num_char_q2'] = test_que2.apply(lambda x: len(x))

# Amount of words in each questions
X_train['num_words_q1'] = train_que1.apply(lambda x: len(x.split(' ')))
X_train['num_words_q2'] = train_que2.apply(lambda x: len(x.split(' ')))

X_test['num_words_q1'] = test_que1.apply(lambda x:len(x.split(' '))) 
X_test['num_words_q2'] = test_que2.apply(lambda x: len(x.split(' ')))

### my new features ###

tr = train_data
tr = tr.fillna('0')

te = test_data
te = te.fillna('0')

# del tr, te

# How much, How long, What, When, Where, Why, Who


X_train['howmuch_q1'] = tr['question1'].apply(lambda x: 'How much ' in x).astype(int)
X_train['howmuch_q2'] = tr['question2'].apply(lambda x: 'How much ' in x).astype(int)
X_test['howmuch_q1'] = te['question1'].apply(lambda x: 'How much ' in x).astype(int)
X_test['howmuch_q2'] = te['question2'].apply(lambda x: 'How much ' in x).astype(int)


X_train['howlong_q1'] = tr['question1'].apply(lambda x: 'How long ' in x).astype(int)
X_train['howlong_q2'] = tr['question2'].apply(lambda x: 'How long ' in x).astype(int)
X_test['howlong_q1'] = te['question1'].apply(lambda x: 'How long ' in x).astype(int)
X_test['howlong_q2'] = te['question2'].apply(lambda x: 'How long ' in x).astype(int)



X_train['what_q1'] = tr['question1'].apply(lambda x: 'What ' in x).astype(int)
X_train['what_q2'] = tr['question2'].apply(lambda x: 'What ' in x).astype(int)
X_test['what_q1'] = te['question1'].apply(lambda x: 'What ' in x).astype(int)
X_test['what_q2'] = te['question2'].apply(lambda x: 'What ' in x).astype(int)


X_train['when_q1'] = tr['question1'].apply(lambda x: 'When ' in x).astype(int)
X_train['when_q2'] = tr['question2'].apply(lambda x: 'When ' in x).astype(int)
X_test['when_q1'] = te['question1'].apply(lambda x: 'When ' in x).astype(int)
X_test['when_q2'] = te['question2'].apply(lambda x: 'When ' in x).astype(int)



X_train['where_q1'] = tr['question1'].apply(lambda x: 'Where ' in x).astype(int)
X_train['where_q2'] = tr['question2'].apply(lambda x: 'Where ' in x).astype(int)
X_test['where_q1'] = te['question1'].apply(lambda x: 'Where ' in x).astype(int)
X_test['where_q2'] = te['question2'].apply(lambda x: 'Where ' in x).astype(int)


X_train['why_q1'] = tr['question1'].apply(lambda x: 'Why ' in x).astype(int)
X_train['why_q2'] = tr['question2'].apply(lambda x: 'Why ' in x).astype(int)
X_test['why_q1'] = te['question1'].apply(lambda x: 'Why ' in x).astype(int)
X_test['why_q2'] = te['question2'].apply(lambda x: 'Why ' in x).astype(int)


X_train['who_q1'] = tr['question1'].apply(lambda x: 'Who ' in x).astype(int)
X_train['who_q2'] = tr['question2'].apply(lambda x: 'Who ' in x).astype(int)
X_test['who_q1'] = te['question1'].apply(lambda x: 'Who ' in x).astype(int)
X_test['who_q2'] = te['question2'].apply(lambda x: 'Who ' in x).astype(int)


X_train['math_q1'] = tr['question1'].apply(lambda x: '[math]' in x).astype(int)
X_train['math_q2'] = tr['question2'].apply(lambda x: '[math]' in x).astype(int)
X_test['math_q1'] = te['question1'].apply(lambda x: '[math]' in x).astype(int)
X_test['math_q2'] = te['question2'].apply(lambda x: '[math]' in x).astype(int)


X_train['fullstop_q1'] = tr['question1'].apply(lambda x: '.' in x).astype(int)
X_train['fullstop_q2'] = tr['question2'].apply(lambda x:  '.' in x).astype(int)
X_test['fullstop_q1'] = te['question1'].apply(lambda x:  '.' in x).astype(int)
X_test['fullstop_q2'] = te['question2'].apply(lambda x:  '.' in x).astype(int)


X_train['diff_q1'] = tr['question1'].apply(lambda x: 'difference' in x).astype(int)
X_train['diff_q2'] = tr['question2'].apply(lambda x:  'difference' in x).astype(int)
X_test['diff_q1'] = te['question1'].apply(lambda x:  'difference' in x).astype(int)
X_test['diff_q2'] = te['question2'].apply(lambda x:  'difference' in x).astype(int)


X_train['bestway_q1'] = tr['question1'].apply(lambda x: 'best way ' in x).astype(int)
X_train['bestway_q2'] = tr['question2'].apply(lambda x:  'best way ' in x).astype(int)
X_test['bestway_q1'] = te['question1'].apply(lambda x:  'best way ' in x).astype(int)
X_test['bestway_q2'] = te['question2'].apply(lambda x:  'best way ' in x).astype(int)



tr.isnull().sum()

te.isnull().sum()



###  MAGIC FEATURE #3         ###

# 1. Building QIDs for Test dataset

f_train=os.path.join(train_csv)
f_test=os.path.join(test_csv)

train_orig =  pd.read_csv(train_csv, header=0)
test_orig =  pd.read_csv(test_csv, header=0)

# "id","qid1","qid2","question1","question2","is_duplicate"
df_id1 = train_orig[["qid1", "question1"]].drop_duplicates(keep="first").copy().reset_index(drop=True)
df_id2 = train_orig[["qid2", "question2"]].drop_duplicates(keep="first").copy().reset_index(drop=True)

df_id1.columns = ["qid", "question"]
df_id2.columns = ["qid", "question"]

print(df_id1.shape, df_id2.shape)

df_id = pd.concat([df_id1, df_id2]).drop_duplicates(keep="first").reset_index(drop=True)
print(df_id1.shape, df_id2.shape, df_id.shape)
#
import csv
dict_questions = df_id.set_index('question').to_dict()
dict_questions = dict_questions["qid"]

new_id = 538000 # df_id["qid"].max() ==> 537933

def get_id(question):
    global dict_questions 
    global new_id 
    
    if question in dict_questions:
        return dict_questions[question]
    else:
        new_id += 1
        dict_questions[question] = new_id
        return new_id
    
rows = []
max_lines = 10
if True:
    with open(f_test, 'r', encoding="utf8") as infile:
        reader = csv.reader(infile, delimiter=",")
        header = next(reader)
        header.append('qid1')
        header.append('qid2')
        
        if True:
            print(header)
            pos, max_lines = 0, 10*1000*1000
            for row in reader:
                # "test_id","question1","question2"
                question1 = row[1]
                question2 = row[2]

                qid1 = get_id(question1)
                qid2 = get_id(question2)
                row.append(qid1)
                row.append(qid2)

                pos += 1
                if pos >= max_lines:
                    break
                rows.append(row)
                

# 2. Calculate Kscore 
### Kcore decomposition ###

df_train = pd.read_csv(train_csv, usecols=["qid1", "qid2"])

df_test = pd.DataFrame(rows)
df_test.columns = ['id','question1','question2','qid1','qid2']


# del df_train, df_test
# del df,df1,df2,df1_test,df2_test,df_all

# df_test.head(10)

# df_test = pd.read_csv(test_csv, usecols=["qid1", "qid2"])

df_all = pd.concat([df_train, df_test])

print("df_all.shape:", df_all.shape) # df_all.shape: (2750086, 2)

df = df_all

import networkx as nx

g = nx.Graph()

g.add_nodes_from(df.qid1)

edges = list(df[['qid1', 'qid2']].to_records(index=False))

g.add_edges_from(edges)

g.remove_edges_from(g.selfloop_edges())

print(len(set(df.qid1)), g.number_of_nodes()) # 4789604

print(len(df), g.number_of_edges()) # 2743365 (after self-edges)

df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])

print("df_output.shape:", df_output.shape)

NB_CORES = 20

for k in range(2, NB_CORES + 1):

    fieldname = "kcore{}".format(k)

    print("fieldname = ", fieldname)

    ck = nx.k_core(g, k=k).nodes()

    print("len(ck) = ", len(ck))

    df_output[fieldname] = 0

    df_output.ix[df_output.qid.isin(ck), fieldname] = k

df_output.to_csv("question_kcores.csv", index=None)



# def run_kcore_max():

df_cores = pd.read_csv("question_kcores.csv", index_col="qid")
# df_cores.head(100)

df_cores.index.names = ["qid"]

df_cores['max_kcore'] = df_cores.apply(lambda row: max(row), axis=1)

df_cores[['max_kcore']].to_csv("question_max_kcores.csv") # with index



cores_dict = pd.read_csv("question_max_kcores.csv", index_col="qid").to_dict()["max_kcore"]

def gen_qid1_max_kcore(row):
    return cores_dict[row["qid1"]]
def gen_qid2_max_kcore(row):
    return cores_dict[row["qid2"]]

#def gen_max_kcore(row):
#    return max(row["qid1_max_kcore"], row["qid2_max_kcore"])


# 2 new features #
X_train["qid1_max_kcore"] = df_train.apply(gen_qid1_max_kcore, axis=1)
X_test["qid1_max_kcore"] = df_test.apply(gen_qid1_max_kcore, axis=1)
X_train["qid2_max_kcore"] = df_train.apply(gen_qid2_max_kcore, axis=1)
X_test["qid2_max_kcore"] = df_test.apply(gen_qid2_max_kcore, axis=1)                





################################33

def coincidence(row):
    word = 0
    while word < len(train_data):
        if train_data['question1'].apply(lambda x: row in x) == 1 and train_data['question2'].apply(lambda x: row in x) == 1:
            return 1
        word = word+1
    return 0    


X_train['how'] = train_data.apply(coincidence('How '), axis =1, raw = True)      
            
##############################################


### This method didn't get a boost ###

### Bag of words ###

 maxNumFeatures = 1000000

# bag of letter sequences (chars)
 BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=50, max_features=maxNumFeatures, 
                                      analyzer='char', ngram_range=(1,10), 
                                      binary=True, lowercase=True)

 trainDF = train_data.dropna(how="any").reset_index(drop=True) 

# trainDF = train_data.fillna('')
# train_data.isnull().sum()

trainDF.isnull().sum()

BagOfWordsExtractor.fit(pd.concat((trainDF.question1,trainDF.question2)).unique())

# BagOfWordsExtractor.fit(pd.concat((trainDF.ix[:,'question1'],trainDF.ix[:,'question2'])).unique())

# trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:,'question1'])
# trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:,'question2'])

question1 = BagOfWordsExtractor.transform(trainDF['question1'])
question2 = BagOfWordsExtractor.transform(trainDF['question2'])

X_featextr = -(question1 != question2).astype(int)
y = trainDF['is_duplicate'].values



logisticRegressor = linear_model.LogisticRegression(C=0.1, solver='sag', 
                                                    class_weight={1: 0.472008228977, 0: 1.30905513329})
logisticRegressor.fit(X_featextr, y)


test_data.isnull().sum()

test_data.ix[test_data['question1'].isnull(),['question1','question2']] = ' '
test_data.ix[test_data['question2'].isnull(),['question1','question2']] = ' '


Question1_test = BagOfWordsExtractor.transform(test_data['question1'])
Question2_test = BagOfWordsExtractor.transform(test_data['question2'])

X_test = -(Question1 != Question2).astype(int)

### WASTED ###





### MAGIC FEATURE #1 ###

# Add Magic Feature #
df1 = train_data[['question1']].copy()
df2 = train_data[['question2']].copy()
df1_test = test_data[['question1']].copy()
df2_test = test_data[['question2']].copy()


df2.rename(columns = {'question2':'question1'},inplace=True)
df2_test.rename(columns = {'question2':'question1'},inplace=True)

train_questions = df1.append(df2)
train_questions = train_questions.append(df1_test)
train_questions = train_questions.append(df2_test)
#train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
train_questions.drop_duplicates(subset = ['question1'],inplace=True)


train_questions.reset_index(inplace=True,drop=True)
questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
train_cp = train_data.copy()
test_cp = test_data.copy()
train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

test_cp['is_duplicate'] = -1
test_cp.rename(columns={'test_id':'id'},inplace=True)
comb = pd.concat([train_cp,test_cp])

comb['q1_hash'] = comb['question1'].map(questions_dict)
comb['q2_hash'] = comb['question2'].map(questions_dict)

q1_vc = comb.q1_hash.value_counts().to_dict()
q2_vc = comb.q2_hash.value_counts().to_dict()


def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0
#map to frequency space
comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_hash','q2_hash','q1_freq','q2_freq','is_duplicate']]
test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq','q2_freq']]



X_train['q1_freq'] = train_comb['q1_freq']
X_train['q2_freq'] = train_comb['q2_freq']


X_test['q1_freq'] = test_comb['q1_freq']
X_test['q2_freq'] = test_comb['q2_freq']



##### MAGIC FEATURE #2 ######

ques = pd.concat([train_data[['question1', 'question2']], 
        test_data[['question1', 'question2']]], axis=0).reset_index(drop='index')
# ques.shape

q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))



X_train['q1_q2_intersect'] = train_data.apply(q1_q2_intersect, axis=1, raw=True)
X_test['q1_q2_intersect'] = test_data.apply(q1_q2_intersect, axis=1, raw=True)















####### Calibrate our target #########

 pos_train = X_train[Y_train == 1]
 neg_train = X_train[Y_train == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
 p = 0.165
 scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
 while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
 neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
 print(len(pos_train) / (len(pos_train) + len(neg_train)))

 X_train_fix = pd.concat([pos_train, neg_train])
 Y_train_fix = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
 del pos_train, neg_train






## RUN XGBoost ##


# Finally, we split some of the data off for validation
from sklearn.cross_validation import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(X_train_fix, Y_train_fix, test_size=0.3, random_state=123)



# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.015 # 0.02
params['max_depth'] = 6
#params['subsample'] = 0.7
#params['colsample_bylevel'] = 0.7
# params['max_delta_step'] = 0.36
#params['scale_pos_weight'] = 0.360

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=50, verbose_eval=10)


## Make predictions ##

d_test = xgb.DMatrix(X_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = test_data['test_id']
sub['is_duplicate'] = p_test

sub.to_csv('quora_newfeat_v5.0.csv', index=False)

# sub.head(10)



# 1. Val = 0.2036; eta=0.015; LB = 0.1733
# 2. Val = 0.1987; Train = 0.1918








### StackNet solution for Quora ###

    from scipy.sparse import csr_matrix,hstack

    train_mix = (train_data['question1']+ " " +  train_data['question2']).astype(str).values
    test_mix = (test_data['question1']+ " " +  test_data['question2'] ).astype(str).values  
    
    
        #convert to csr
    X_stack=csr_matrix(X_train)
    X_test_stack=csr_matrix(X_test)
    # the tfidf object
    tfidf=TfidfVectorizer(min_df=1, max_features=None, strip_accents='unicode',lowercase =True,
                        analyzer='word', token_pattern=r'\w{2,}', ngram_range=(1, 1), use_idf=True,smooth_idf=True, 
    sublinear_tf=True, stop_words = 'english')  
    
    # aplied tf-idf
    tr_sparsed  = tfidf. fit_transform (train_mix)  
    te_sparsed = tfidf.transform(test_mix)
    print (tr_sparsed.shape, te_sparsed.shape, X_stack.shape, X_test_stack.shape)  

    #join the the tfidf with the remaining data
    X_stack =hstack([X_stack,tr_sparsed]).tocsr()#
    X_test_stack = hstack([X_test_stack, te_sparsed]).tocsr()#

    #retrieve target
    y = train_data['is_duplicate'].values  
    print (X_stack.shape, X_test_stack.shape, y.shape) 
    
    
def fromsparsetofile(filename, array, deli1=" ", deli2=":",ytarget=None):    
    zsparse=csr_matrix(array)
    indptr = zsparse.indptr
    indices = zsparse.indices
    data = zsparse.data
    print(" data lenth %d" % (len(data)))
    print(" indices lenth %d" % (len(indices)))    
    print(" indptr lenth %d" % (len(indptr)))
    
    f=open(filename,"w")
    counter_row=0
    for b in range(0,len(indptr)-1):
        #if there is a target, print it else , print nothing
        if ytarget!=None:
             f.write(str(ytarget[b]) + deli1)     
             
        for k in range(indptr[b],indptr[b+1]):
            if (k==indptr[b]):
                if np.isnan(data[k]):
                    f.write("%d%s%f" % (indices[k],deli2,-1))
                else :
                    f.write("%d%s%f" % (indices[k],deli2,data[k]))                    
            else :
                if np.isnan(data[k]):
                     f.write("%s%d%s%f" % (deli1,indices[k],deli2,-1))  
                else :
                    f.write("%s%d%s%f" % (deli1,indices[k],deli2,data[k]))
        f.write("\n")
        counter_row+=1
        if counter_row%10000==0:    
            print(" row : %d " % (counter_row))    
    f.close()  
    
    
    
    
    
    
    #export sparse data to stacknet format (which is Libsvm format)
    fromsparsetofile("train.sparse", X_stack, deli1=" ", deli2=":",ytarget=y)    
    fromsparsetofile("test.sparse", X_test_stack, deli1=" ", deli2=":",ytarget=None)       
    














#########################################

################ StackNet solution #################################3#####


#create average value of the target variabe given a categorical feature        
def convert_dataset_to_avg(xc,yc,xt, rounding=2,cols=None):
    xc=xc.tolist()
    xt=xt.tolist()
    yc=yc.tolist()
    if cols==None:
        cols=[k for k in range(0,len(xc[0]))]
    woe=[ [0.0 for k in range(0,len(cols))] for g in range(0,len(xt))]
    good=[]
    bads=[]
    for col in cols:
        dictsgoouds=defaultdict(int)        
        dictsbads=defaultdict(int)
        good.append(dictsgoouds)
        bads.append(dictsbads)        
    total_count=0.0
    total_sum =0.0

    for a in range (0,len(xc)):
        target=yc[a]
        total_sum+=target
        total_count+=1.0
        for j in range(0,len(cols)):
            col=cols[j]
            good[j][round(xc[a][col],rounding)]+=target
            bads[j][round(xc[a][col],rounding)]+=1.0  
    #print(total_goods,total_bads)            
    
    for a in range (0,len(xt)):    
        for j in range(0,len(cols)):
            col=cols[j]
            if round(xt[a][col],rounding) in good[j]:
                 woe[a][j]=float(good[j][round(xt[a][col],rounding)])/float(bads[j][round(xt[a][col],rounding)])  
            else :
                 woe[a][j]=round(total_sum/total_count)
    return woe            
    

#converts the select categorical features to numerical via creating averages based on the target variable within kfold. 

def convert_to_avg(X,y, Xt, seed=1, cvals=5, roundings=2, columns=None):
    
    if columns==None:
        columns=[k for k in range(0,(X.shape[1]))]    
    #print("it is not!!")        
    X=X.tolist()
    Xt=Xt.tolist() 
    woetrain=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(X))]
    woetest=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(Xt))]    
    
    kfolder=StratifiedKFold(y, n_folds=cvals,shuffle=True, random_state=seed)
    for train_index, test_index in kfolder:
        # creaning and validation sets
        X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
        y_train = np.array(y)[train_index]

        woecv=convert_dataset_to_avg(X_train,y_train,X_cv, rounding=roundings,cols=columns)
        X_cv=X_cv.tolist()
        no=0
        for real_index in test_index:
            for j in range(0,len(X_cv[0])):
                woetrain[real_index][j]=X_cv[no][j]
            no+=1
        no=0
        for real_index in test_index:
            for j in range(0,len(columns)):
                col=columns[j]
                woetrain[real_index][col]=woecv[no][j]
            no+=1      
    woefinal=convert_dataset_to_avg(np.array(X),np.array(y),np.array(Xt), rounding=roundings,cols=columns) 

    for real_index in range(0,len(Xt)):
        for j in range(0,len(Xt[0])):           
            woetest[real_index][j]=Xt[real_index][j]
            
    for real_index in range(0,len(Xt)):
        for j in range(0,len(columns)):
            col=columns[j]
            woetest[real_index][col]=woefinal[real_index][j]
            
    return np.array(woetrain), np.array(woetest)

        


        train_file="train_stacknet.csv"
        test_file="test_stacknet.csv"
        
        ######### Load files ############

  #      X,X_test,y,ids=load_data_sparse (data_path="input/")# you might need to change that to whatever folder the json files are in
  #      ids= np.array([int(k)+68119576 for k in ids ]) # we add the id value we removed before for scaling reasons.

        #create to numpy arrays (dense format)    
        
        X_train_na = X_train_fix.fillna(0)
        X_test_na = X_test.fillna(0)
    
        X= X_train_na.as_matrix()
        X_test_new= X_test_na.as_matrix()  
        
        y = Y_train_fix
        yy =y.as_matrix()
        ids = test_data['test_id']
        
        print ("scalling") 
        #scale the data
        stda=StandardScaler()  
        X_test_new = stda.fit_transform (X_test_new)          
        X=stda.transform(X)

        
  #      CO=[0,14,21] # columns to create averages on
        CO = [1]
        #Create Arrays for meta
        train_stacker=[ [0.0 for s in range(2)]  for k in range (0,(X.shape[0])) ]
        test_stacker=[[0.0 for s in range(2)]   for k in range (0,(X_test_new.shape[0]))]
        
        numf = 5 # number of folds to use
        print("kfolder")
        #cerate 5 fold object
        mean_logloss = 0.0
        kfolder=StratifiedKFold(y, n_folds = numf,shuffle=True, random_state=123)   # Change on KFold

        #xgboost_params
        param = {}
        param['booster']='gbtree'
        param['objective'] = 'multi:softprob'
        param['bst:eta'] = 0.02 # 0.04
        param['seed']=  1
        param['bst:max_depth'] = 6
        param['bst:min_child_weight']= 1.
        param['silent'] =  1  
        param['nthread'] = 12 # put more if you have
        param['bst:subsample'] = 0.7
        param['gamma'] = 1.0
        param['colsample_bytree']= 1.0
   #     param['num_parallel_tree']= 3   
        param['colsample_bylevel']= 0.7                  
        param['lambda']=5  
        param['num_class']= 2 # must be '1' here, binary target





        i=0 # iterator counter
        print ("starting cross validation with %d kfolds " % (number_of_folds))
        for train_index, test_index in kfolder:
                # creaning and validation sets
                X_train, X_cv = X[train_index], X[test_index]
                y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
                #create past averages for some fetaures
                W_train,W_cv=convert_to_avg(X_train,y_train, X_cv, seed=1, cvals=5, roundings=2, columns=CO)
                W_train=np.column_stack((X_train,W_train[:,CO]))
                W_cv=np.column_stack((X_cv,W_cv[:,CO])) 
                print (" train size: %d. test size: %d, cols: %d " % ((W_train.shape[0]) ,(W_cv.shape[0]) ,(W_train.shape[1]) ))
                #training
                X1=xgb.DMatrix(csr_matrix(W_train), label=np.array(y_train),missing =-999.0)
                X1cv=xgb.DMatrix(csr_matrix(W_cv), missing =-999.0)
                bst = xgb.train(param.items(), X1, 1000) 
                #predictions
                predictions = bst.predict(X1cv)     
                preds=predictions.reshape( W_cv.shape[0], 2) # binary target

                #scalepreds(preds)     
                logs = log_loss(y_cv,preds)
                print ("size train: %d size cv: %d loglikelihood (fold %d/%d): %f" % ((W_train.shape[0]), (W_cv.shape[0]), i + 1, number_of_folds, logs))
             
                mean_logloss += logs
                #save the results
                no=0
                for real_index in test_index:
                    for d in range (0,2):
                        train_stacker[real_index][d]=(preds[no][d])
                    no+=1
                i+=1
        mean_logloss/=number_of_folds
        print (" Average Lolikelihood: %f" % (mean_logloss) )
                
        #calculating averages for the train data
        W,W_test=convert_to_avg(X,y, X_test_new, seed=2, cvals=5, roundings=2, columns=CO)
        W=np.column_stack((X,W[:,CO]))
        W_test=np.column_stack((X_test_new,W_test[:,CO]))          
        #X_test=np.column_stack((X_test,woe_cv))      
        print (" making test predictions ")

        X1=xgb.DMatrix(csr_matrix(W), label=np.array(y) , missing =-999.0)
        X1cv=xgb.DMatrix(csr_matrix(W_test), missing =-999.0)
     
        bst = xgb.train(param.items(), X1, 1000) 
  
        predictions = bst.predict(X1cv)     
        preds=predictions.reshape( W_test.shape[0], 2)        
       
        for pr in range (0,len(preds)):  
                for d in range (0,2):            
                    test_stacker[pr][d]=(preds[pr][d]) 
        
        
        
        print ("merging columns")   
        #stack xgboost predictions
        X_stacknet = np.column_stack((X,train_stacker))
        # stack id to test
        X_test_stacknet = np.column_stack((X_test_new,test_stacker))        
        
        # stack target to train
        X_stacknet = np.column_stack((y,X_stacknet))
        # stack id to test
        X_test_stacknet = np.column_stack((ids,X_test_stacknet))
        
        #export to txt files (, del.)
        print ("exporting files")
        np.savetxt(train_file, X_stacknet, delimiter=",", fmt='%.5f')
        np.savetxt(test_file, X_test_stacknet, delimiter=",", fmt='%.5f')   


# Load Stacknet predictions #

    stacknet_pr = pd.read_csv('/home/pavel/anaconda3/Scripts/Quora/Submissions/sigma_stack_pred.csv',header=None)

    sub_new = pd.DataFrame()

    sub_new['test_id'] = test_data['test_id']
    sub_new['is_duplicate'] = stacknet_pr.ix[:,1]
    
    sub_new.isnull().sum()

    sub_new.to_csv('quora_stacknet_v3.0.csv',index=False)






### HyperOpt searching best params ###
Y_train_new = pd.DataFrame()

Y_train_new['is_duplicate'] = Y_train_fix

X_train_rf.isnull().sum()
X_train_rf = X_train_fix.fillna(0)

 from sklearn.model_selection import KFold, StratifiedKFold

def objective(space):
    
    numfolds = 5
    total = 0
    kf = StratifiedKFold(n_splits=numfolds, shuffle=True,random_state=13)
    
#        param = {}
#        param['booster']='gbtree'
#        param['objective'] = 'multi:softprob'
#        param['bst:eta'] = 0.02 # 0.04
#        param['seed']=  1
#        param['bst:max_depth'] = 6
#        param['bst:min_child_weight']= 1.
#        param['silent'] =  1  
#        param['nthread'] = 12 # put more if you have
#        param['bst:subsample'] = 0.7
#        param['gamma'] = 1.0
#        param['colsample_bytree']= 1.0
   #     param['num_parallel_tree']= 3   
#        param['colsample_bylevel']= 0.7                  
#        param['lambda']=5      
    
    
    
    
    
    clf = xgb.XGBClassifier(n_estimators = 1000, 
                            max_depth = space['max_depth'],
                            learning_rate = space['learning_rate'],
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                            colsample_bytree = space['colsample_bytree'],
                            colsample_bylevel = space['colsample_bylevel'],
                            lambd = space['lambda'],
                            nthread = 12)
    
#    rf = RandomForestClassifier(n_estimators = 1000, 
#                            max_depth = space['max_depth'],
#                            max_features = space['max_features'],
#                            criterion = space['criterion'],
#                            min_impurity_split = space['min_impurity_split'],
             #               scale = space['scale'],
             #               normalize = space['normalize'],
             #               min_samples_leaf = space['min_samples_leaf'],
             #               min_weight_fraction_leaf  = space['min_weight_fraction_leaf'],
             #               min_impurity_split = space['min_impurity_split'],
                            random_state = 13,
                            warm_start = True,                            
                            n_jobs = -1
                            )
    
    for train_index, test_index in kf.split(X_train_fix,Y_train_new.is_duplicate):
        xtrain, xtest = X_train_rf.iloc[train_index], X_train_rf.iloc[test_index]
        ytrain, ytest = Y_train_new.iloc[train_index], Y_train_new.iloc[test_index]
        
        eval_set = [(xtrain, ytrain),(xtest, ytest)]

        clf.fit(xtrain, ytrain, eval_metric="logloss",eval_set = eval_set, early_stopping_rounds=50)
 
#        rf.fit(xtrain,ytrain.values.ravel())
        pred = clf.predict_proba(xtest)[:,1]
     
        logloss = log_loss(ytest, pred)
        print ("SCORE:", logloss)  
        total += logloss
    total = total/numfolds
    print (total)
    return{'loss':total, 'status': STATUS_OK }


space ={
    'max_depth': hp.choice('max_depth', range(1,10)),
#    'max_features': hp.choice('max_features', range(1,8)),
#   'n_estimators': hp.choice('n_estimators', range(1,50)),
#    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'min_impurity_split': hp.uniform('min_impurity_split',0,0.3)                       
 #   'scale': hp.choice('scale', [0, 1]),
#    'normalize': hp.choice('normalize', [0, 1])
    }


#space ={
#        'max_depth': hp.choice('max_depth', np.arange(1, 10, dtype=int)),
#        'learning_rate': hp.quniform('learning_rate', 0, 0.03, 0.01),
#        'min_child_weight': hp.quniform ('min_child_weight', 1, 10, 1),
#        'subsample': hp.uniform ('x_subsample', 0.7, 1)
#    }


  
    
trials = Trials()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10, 
            trials=trials) 

print (best)   

trials.results
trials.trials


## CV = 0.25848; LB = 0.2291;

    numfolds = 5
    total = 0
    kf = StratifiedKFold(n_splits=numfolds, shuffle=True,random_state=13)
    
#    clf = xgb.XGBClassifier(n_estimators = 10000, 
#                            max_depth = 5,
#                            learning_rate = 0.01,
#                            min_child_weight = 7,
#                            subsample = 0.899431)
    rf = RandomForestClassifier(n_estimators = 1000, 
                            max_depth = 8,
                            max_features = 2,
                            criterion = "gini",
                            min_impurity_split = 0.005356707662170046,
             #               scale = space['scale'],
             #               normalize = space['normalize'],
             #               min_samples_leaf = space['min_samples_leaf'],
             #               min_weight_fraction_leaf  = space['min_weight_fraction_leaf'],
             #               min_impurity_split = space['min_impurity_split'],
                            random_state = 13,
                            warm_start = True,                            
                            n_jobs = -1
                            )
 



   
    for train_index, test_index in kf.split(X_train_rf,Y_train_new.is_duplicate):
        xtrain, xtest = X_train_rf.iloc[train_index], X_train_rf.iloc[test_index]
        ytrain, ytest = Y_train_new.iloc[train_index], Y_train_new.iloc[test_index]
        
        eval_set = [(xtrain, ytrain),(xtest, ytest)]

        rf.fit(xtrain, ytrain)
        pred = rf.predict_proba(xtest)[:,1]
     
        logloss = log_loss(ytest, pred)
#        print ("SCORE:", logloss)  
        total += logloss
    total = total/numfolds
    print (total)


## Predictions ###


 Pred_prob = clf.predict_proba(X_test)
 
# Pred_rf = rf.predict_proba(X_test_rf)

X_test_rf = X_test.fillna(0)


Pred_prod_df = pd.DataFrame(Pred_prob)
del Pred_prod_df[0]

sub_new = pd.DataFrame()

sub_new['test_id'] = test_data['test_id']
sub_new['is_duplicate'] = Pred_prod_df


sub_new.to_csv('quora_hyperopt_v2.4.csv',index=False)




### Averaging subs ###

sub2 = pd.read_csv('/home/pavel/anaconda3/Scripts/Quora/Submissions/quora_newfeat_v4.0.csv')

sub_final = pd.DataFrame()

sub_final['test_id'] = sub['test_id']

sub_final['is_duplicate'] = (sub['is_duplicate'] + sub2['is_duplicate'])/2

sub_final.to_csv('quora_avg_newfeat_v4.2.csv',index=False)


