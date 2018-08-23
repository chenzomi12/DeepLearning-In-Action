
# coding: utf-8

# In[1]:


import nltk
import json
import itertools
import numpy as np


# {
#   "asin": "B000050B6Z",
#   "questionType": "yes/no",
#   "answerType": "Y",
#   "answerTime": "Aug 8, 2014",
#   "unixTime": 1407481200,
#   "question": "Can you use this unit with GEL shaving cans?",
#   "answer": "Yes. If the can fits in the machine it will despense hot gel lather. I've been using my machine for both , gel and traditional lather for over 10 years."
# }

# In[2]:


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
tokenized_cent[0]


# In[3]:


word_freq = nltk.FreqDist(itertools.chain(*tokenized_cent)) # 2327
len(word_freq)


# In[4]:


vocabulary_size = 1800
vocab = word_freq.most_common(vocabulary_size)
index_2_word = dict([(i, w[0]) for i,w in enumerate(vocab)])
index_2_word[vocabulary_size] = unknow_token

word_2_index = dict([(index_2_word[i], i) for i,w in enumerate(index_2_word)])


# In[5]:


for i, sent in enumerate(tokenized_cent):
    tokenized_cent[i] = [w if w in word_2_index else unknow_token for w in sent]


# In[6]:


X_pre_train = np.asarray([[word_2_index[w] for w in sent[:-1]] for sent in tokenized_cent[:800]])
X_pre_text = np.asarray([[word_2_index[w] for w in sent[:-1]] for sent in tokenized_cent[800:1000]])
Y_pre_train = np.asarray([[word_2_index[w] for w in sent[1:]] for sent in tokenized_cent[:800]])
Y_pre_test = np.asarray([[word_2_index[w] for w in sent[1:]] for sent in tokenized_cent[800:1000]])


# In[7]:


X_pre_train.shape
len(word_2_index)


# In[8]:


X_train = np.eye(vocabulary_size+1)[X_pre_train]
X_test = np.eye(vocabulary_size+1)[X_pre_text]
Y_train = np.eye(vocabulary_size+1)[Y_pre_train]
Y_test = np.eye(vocabulary_size+1)[Y_pre_test]


# In[9]:


X_train[1]


# In[10]:


X_train.shape[2]


# In[11]:


class SmapleRNN:
    def __init__(self, input_dim, hidden_dim=50):
        # 设置RNN网络的输入、输出、隐层的维度
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 定义RNN网络权重大小
        U_size = (self.hidden_dim, self.input_dim)
        W_size = (self.hidden_dim, self.hidden_dim)
        V_size = (self.input_dim, self.hidden_dim)
        
        # 随机初始化RNN网络权重参数
        random_min = -np.sqrt(1./self.input_dim)
        random_max = np.sqrt(1./self.input_dim)
        self.U = np.random.uniform(random_min, random_max, U_size)
        self.W = np.random.uniform(random_min, random_max, W_size)
        self.V = np.random.uniform(random_min, random_max, V_size)
        
    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def _tanh(self, x):
        """Sample translate the input data to numpy tanh function."""
        return np.tanh(x)
    
    def forward_propagation(self, x):
        """向前传播算法"""
        time = x.shape[0]    # 获取输入数据中的序列数time

        s = np.zeros((time+1, self.hidden_dim)) # (15, 50) 定义状态矩阵
        h = np.zeros((time, self.input_dim))  # (14, 1801) 定义输出
        
        # 根据公式计算隐层状态  和输出  的值
        for t in np.arange(time):
            s[t] = self._tanh(np.dot(self.U, x[t]) + np.dot(self.W, s[t-1]))
            h[t] = self._softmax(self.V.dot(s[t]))

        return (s, h)
    
    def predict(self, x):
        """根据输入序列数据预测序列输出"""
        s, h = self.forward_propagation(x)
        
        print("input shape:{}.".format(x.shape))    #>>> input shape:(14, 1801).
        print("status shape:{}.".format(s.shape))    #>>> status shape:(15, 50).
        print("output shape:{}.".format(h.shape))    #>>> output shape:(14, 1801).
        
        return np.argmax(h, axis=1)
    
    def calc_loss(self, x, y):
        """计算损失""" 
        loss = 0
        time = y.shape[0]
        
        s, y_predict = self.forward_propagation(x)

        # t时间上的损失
        loss_t = np.sum(y*(np.log(y_predict)), axis=1)
        
        # 对单个时间序列上的损失求和，然后求平均
        total_loss = - np.sum(loss_t) / time
        
        return total_loss
    
    def backward_propagation_though_time(self, x, y):
        bptt_truncate = 4
        
        time = y.shape[0]
        
        s, y_predict = SampleRNN.forward_propagation(x)
        
        dE_dV = np.zeros_like(self.V)
        dE_dW = np.zeros_like(self.W)
        dE_dU = np.zeros_like(self.U)
        
        dE_dy = y - y_predict
        
        for t in reversed(range(time)):
            dE_dV += np.outer(dE_dy[t], s[t].T)
            
            # 首先计算 delta t，在第一次计算的时候 t = 3 
            delta_k = (self.V * dE_dy) * (1 - pow(s[t], 2))
            
            # 开始BPTT步骤 
            for step_t in reversed(arange(t - bptt_truncate, t + 1)):
                dE_dW += np.outer(delta_k, s[step_t-1]) # 加到之前每一步的梯度上
                dE_dU[:, x[step_t]] += delta_k
                delta_k = (self.W * delta_k) * (1 - pow(s[step_t-1], 2))


# In[12]:


rnn_model = SmapleRNN(vocabulary_size + 1)
y_predict = rnn_model.predict(X_train[100])
rnn_model.calc_loss(X_train[100], Y_train[100])


# In[13]:


y_predict.shape
y_words = []


# In[14]:


y_predict_sent = [index_2_word[i] for i in y_predict]
y_predict_sent = ' '.join(y_predict_sent)
y_predict_sent


# In[15]:


for a in np.arange(max(0, 14-4), 14+1)[::-1]:
    print(a)


# In[16]:


X_train[100].shape
np.argmax(X_train[100], axis=1)


# In[17]:


y = Y_train[100]
np.argmax(Y_train[100], axis=1)
T=len(y)


# In[18]:


14*np.log(1800)


# In[19]:


def gradients_clipping(g, th = 0.8):
    """梯度截断：根据给定的阈值在每一次求得导数后进行判别"""
    if( g > th ):
        g = th/abs(g) * g # abs(g) 取导数的绝对值
    return g

