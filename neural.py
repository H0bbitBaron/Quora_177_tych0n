'''
Example of an LSTM model with GloVe embeddings along with magic features

Tested under Keras 2.0 with Tensorflow 1.0 backend

Single model may achieve LB scores at around 0.18+, average ensembles can get 0.17+
'''

########################################
## import packages
########################################

#import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import gc

#from string import punctuation
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
np.random.seed(519654654)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout#, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.utils import plot_model

from sklearn.preprocessing import StandardScaler

import en_core_web_sm
nlp = en_core_web_sm.load()

########################################
## set directories and parameters
########################################
BASE_DIR = 'Data/'
EMBEDDING_FILE = BASE_DIR + 'glove.840B.300d.txt'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
TRAIN_DATA_FILE_ADD = BASE_DIR + 'train_additional.csv'
TEST_DATA_FILE_ADD = BASE_DIR + 'test_additional.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set
clean_steps = True # delete variables after usage

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

########################################
## index word vectors
########################################
print('Indexing word vectors')

embeddings_index = {} # define a dictionary
f = open(EMBEDDING_FILE,encoding="utf8") # open file with word vectors
count = 0
for line in f: # for each line of file with word vectors
    values = line.split(' ') # split words by space and create dictionary
    word = values[0] # first item in dictionary is word name
    coefs = np.asarray(values[1:], dtype='float32') # other items are vactor array
    embeddings_index[word] = coefs # for each word create list of vector coefs
f.close()

print('Found %d word vectors of glove.' % len(embeddings_index))

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split(' ')

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text) # all others except in list
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

texts_1 = [] 
texts_2 = []
text_similarity = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f: # open file in string stream
    reader = csv.reader(f, delimiter=',') # create link
    header = next(reader) # read first string into header
    for values in reader: # fro each string do
        q1 = nlp(values[3]) # initialise nlp object
        q2 = nlp(values[4])
        text_similarity.append(q1.similarity(q2)) # return similarity between questions
        texts_1.append(text_to_wordlist(values[3])) # parse question1 and replace characters and put into matrix
        texts_2.append(text_to_wordlist(values[4])) # parse question2 and replace characters and put into matrix
        labels.append(int(values[5])) # add target flag
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_text_similarity = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        q1 = nlp(values[1]) # initialise nlp object
        q2 = nlp(values[2])
        test_text_similarity.append(q1.similarity(q2)) # return similarity between questions
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        test_ids.append(values[0])
print('Found %s texts in test.csv' % len(test_texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS) # create one more parsing object with limit of N words
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2) # add text to database

sequences_1 = tokenizer.texts_to_sequences(texts_1) # convert words in each string into freq list
sequences_2 = tokenizer.texts_to_sequences(texts_2)
if clean_steps:
    del texts_1
    del texts_2
    gc.collect()
    
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index # form a word - freq dictionary
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH) # truncate to 30 words and convert to array
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
if clean_steps:
    del sequences_1
    del sequences_2
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
if clean_steps:
    del test_sequences_1
    del test_sequences_2
test_ids = np.array(test_ids)

########################################
## generate leaky features
########################################

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

ques = pd.concat([train_df[['question1', 'question2']], \
        test_df[['question1', 'question2']]], axis=0).reset_index(drop='index') # append test to train
q_dict = defaultdict(set) # create dictionary structure
for i in range(ques.shape[0]): # fill it with question pairs A->B B->A
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])
def q1_freq(row):
    return(len(q_dict[row['question1']]))
def q2_freq(row):
    return(len(q_dict[row['question2']]))    
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))
train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
train_df['q1_freq'] = train_df.apply(q1_freq, axis=1, raw=True)
train_df['q2_freq'] = train_df.apply(q2_freq, axis=1, raw=True)

test_df['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)
test_df['q1_freq'] = test_df.apply(q1_freq, axis=1, raw=True)
test_df['q2_freq'] = test_df.apply(q2_freq, axis=1, raw=True)

leaks = train_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
test_leaks = test_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
if clean_steps:
    del train_df
    del test_df
    del ques
    del q_dict
    gc.collect()

train_df_add=pd.read_csv(TRAIN_DATA_FILE_ADD).drop(['q1_freq','q2_freq','q1_q2_intersect'], axis=1).fillna(0).replace(np.inf, 0)
test_df_add=pd.read_csv(TEST_DATA_FILE_ADD).drop(['q1_freq','q2_freq','q1_q2_intersect'], axis=1).fillna(0).replace(np.inf, 0)

ss = StandardScaler() # create std object
ss.fit(np.vstack((np.hstack((leaks, train_df_add, pd.DataFrame(text_similarity, columns=['simil']))), np.hstack((test_leaks, test_df_add, pd.DataFrame(test_text_similarity, columns=['simil'])))))) # calc stats
leaks = ss.transform(np.hstack((leaks, train_df_add, pd.DataFrame(text_similarity, columns=['simil'])))) # apply std
test_leaks = ss.transform(np.hstack((test_leaks, test_df_add, pd.DataFrame(test_text_similarity, columns=['simil']))))

if clean_steps:
    del train_df_add
    del test_df_add

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1 # get number of words in all files

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM)) # fill matrix with 0
for word, i in word_index.items(): # for each word in freq dict
    embedding_vector = embeddings_index.get(word) # get vectors coefs by word
    if embedding_vector is not None: # if the word is slightly common then save
        embedding_matrix[i] = embedding_vector
if clean_steps:
    del embeddings_index
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################

perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
leaks_train = np.vstack((leaks[idx_train], leaks[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
leaks_val = np.vstack((leaks[idx_val], leaks[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))
if clean_steps:
    del data_1
    del data_2

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words, # number of words with freq in top 200000
        EMBEDDING_DIM,
        weights=[embedding_matrix], # vectors for not null
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

leaks_input = Input(shape=(leaks.shape[1],))
leaks_dense = Dense(num_dense//2, activation=act)(leaks_input)

merged = concatenate([x1, y1, leaks_dense])
merged = BatchNormalization()(merged)
merged = Dropout(rate_drop_dense)(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = BatchNormalization()(merged)
merged = Dropout(rate_drop_dense)(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
# plot_model(model, to_file=BASE_DIR+'model 1.2.png',show_shapes='true')
# model.summary()
# print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = 'Results/' + STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train, leaks_train], labels_train, \
        validation_data=([data_1_val, data_2_val, leaks_val], labels_val, weight_val), \
        epochs=200, batch_size=2048, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2, test_leaks], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1, test_leaks], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('Results/'+'%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
