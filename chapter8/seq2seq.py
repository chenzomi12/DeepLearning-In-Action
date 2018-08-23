
# coding: utf-8

# In[1]:


import spacy
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten, Embedding
from keras.models import Model, Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import RepeatVector, Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed


# In[19]:


nlp = spacy.load('en')
nlp.vocab


# In[24]:


# sequence length; shorter sequences are padded with zeros
MAX_SEQ_LEN = 20
# default from glove embedding vectors
EMBEDDING_DIM = 300
# limits to number of words
MAX_NB_WORDS = 10000
# train/dev split
DEV_SPLIT = 0.1

dropout = 0.5
input_depth = 1
output_depth = 1
input_dim = 128
output_dim = 128
depth = (input_depth, output_depth)
hidden_dim = (input_dim, output_dim)


# In[25]:


def get_embeddings(vocab):
    """
    get embeddings from spacy's glove vectors
    """
    max_rank = MAX_NB_WORDS
    # add 1 to array so we can handle <UNK> words
    vectors = np.ndarray((max_rank + 1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector and lex.rank < MAX_NB_WORDS:
            vectors[lex.rank] = lex.vector
    return vectors

embeddings = get_embeddings(nlp.vocab)
embeddings.shape


# In[67]:


sequence_input = Input(shape=(MAX_SEQ_LEN, ), dtype='int32', name='Input')
print(sequence_input)
sequence_embeded = Embedding(input_dim=embeddings.shape[0], output_dim=EMBEDDING_DIM, 
                             weights=[embeddings], input_length=MAX_SEQ_LEN, 
                             trainable=False, name='Embedding')(sequence_input)
print(sequence_embeded)


# In[68]:


# encoder
encoder = LSTM(hidden_dim[0], return_sequences=True,
               activation='relu', name='Encoder1')(sequence_embeded)
for _ in range(0, depth[0]):
    encoder = LSTM(hidden_dim[0], return_sequences=False,
                   activation='relu', name='Encoder%d' % _)(encoder)
    encoder = Dropout(dropout, name='EnDropout%d' % _)(encoder)

# thought vector
thought_vector = RepeatVector(MAX_SEQ_LEN, name='C')(encoder)

# decoder
decoder = LSTM(hidden_dim[1], return_sequences=True,
               activation='relu', name='Decoder1')(thought_vector)
for _ in range(0, depth[1]):
    decoder = LSTM(hidden_dim[1], return_sequences=True,
                   activation='relu', name='Decoder%d' % _)(decoder)
    decoder = Dropout(dropout, name='DeDropout%d' % _)(decoder)

preds = TimeDistributed(
    Dense(embeddings.shape[0], activation='softmax'))(decoder)


# In[69]:


model = Model(input=sequence_input, output=preds)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

