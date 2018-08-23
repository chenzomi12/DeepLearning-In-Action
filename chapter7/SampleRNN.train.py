
# coding: utf-8

# In[45]:


import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import Dense, Activation, TimeDistributed

import nltk
import json
import itertools


# In[6]:


start_token = "BEGIN"
end_token = "END"
pad_token = "PAD"
unknow_token = "UNKNOW"

with open("Downloads/qa_Appliances_quote.json") as json_file:
    json_data = json.load(json_file)

question_sent = [x['question'].lower() for x in json_data]
tokenized_cent = [nltk.word_tokenize(x)[:13] for x in question_sent]
for i, cent in enumerate(tokenized_cent):
    cent.append(end_token)
    cent.insert(0, start_token)
    while(len(cent)<15):
        cent.append(pad_token)


# In[7]:


word_freq = nltk.FreqDist(itertools.chain(*tokenized_cent)) # 2327
len(word_freq)


# In[9]:


vocabulary_size = 1800
vocab = word_freq.most_common(vocabulary_size)
index_2_word = dict([(i, w[0]) for i,w in enumerate(vocab)])
index_2_word[vocabulary_size] = unknow_token

word_2_index = dict([(index_2_word[i], i) for i,w in enumerate(index_2_word)])

for i, sent in enumerate(tokenized_cent):
    tokenized_cent[i] = [w if w in word_2_index else unknow_token for w in sent]


# In[50]:


X_pre_train = np.asarray([[word_2_index[w] for w in sent[:-1]] for sent in tokenized_cent[:800]])
X_pre_text = np.asarray([[word_2_index[w] for w in sent[:-1]] for sent in tokenized_cent[800:1000]])
Y_pre_train = np.asarray([[word_2_index[w] for w in sent[1:]] for sent in tokenized_cent[:800]])
Y_pre_test = np.asarray([[word_2_index[w] for w in sent[1:]] for sent in tokenized_cent[800:1000]])

X_train = np.eye(vocabulary_size+1)[X_pre_train]
X_test = np.eye(vocabulary_size+1)[X_pre_text]
Y_train = np.eye(vocabulary_size+1)[Y_pre_train]
Y_test = np.eye(vocabulary_size+1)[Y_pre_test]


# In[51]:


# [sample, time steps, features]
# (batch_size, time_step_size, input_vec_size)
print(X_train.shape)
print(Y_train.shape)
print(Y_train.shape[1:])


# In[74]:


# build the model: a signal Simple RNN
print('Build Simple RNN model...')

rnn_model = Sequential()
rnn_model.add(SimpleRNN(256, input_shape=X_train.shape[1:], return_sequences=True, name='RNN'))
rnn_model.add(TimeDistributed(Dense(Y_train.shape[2], activation='softmax'), name='softmax'))

rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(rnn_model.summary())


# In[62]:


model.fit(X_train, Y_train, nb_epoch=30, batch_size=64)
# scores = model.evaluate(X_test, Y_test, verbose=0)
# print("Accuracy:%.2f%%" % (scores[1]*100))


# In[69]:


train_predict = model.predict(X_train)
print(train_predict.shape)

input_yy = np.argmax(Y_train[0], axis=1)
predict_yy = np.argmax(train_predict[0], axis=1)

print(input_yy)
print(predict_yy)

print([index_2_word[i] for i in input_yy])
print([index_2_word[i] for i in predict_yy])


# In[33]:


# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
filename = "/Users/chenzomi/Downloads/min-char-rnn/shakespear.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters:%d"% n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


# In[37]:


print(y.shape)
print(X.shape)

