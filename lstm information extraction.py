import pandas as pd
import numpy as np
import jieba
import re
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

data = pd.read_excel("../警情数据.xls",sheet_name=0)

## 数据预处理
data['警情描述'] = data['警情描述'].str.replace('[^\w]','')
data['tokenize'] = data['警情描述'].apply(jieba.lcut)
data.tokenize = data.tokenize.apply(' '.join)
data_use = data[['tokenize','事故类型']]
data_use.tokenize = data_use.tokenize.str.split(' ')

#model = Word2Vec(text_list, size=100, window=5, min_count=50, workers=4)
model = Word2Vec.load('word2vec.model') # 使用已经训练好的word2vec模型

data_test = data_use[0:400]

voc_list = list(model.wv.vocab.keys()) # 模型中所有涵盖的词语

def choose_token(token):
    token_tmp = [value for value in token if value in voc_list]
    return token_tmp

data_test.tokenize = data_test.tokenize.apply(choose_token) # 找出训练集中模型不涵盖的词语并删除

texts_index = [] 

for i, text in enumerate(data_test.tokenize):
    list_temp = []
    for t, word in enumerate(text):
        try:
            temp = model.wv.vocab[word].index#把每个文档中的每个词，都转为index表示
            
        except KeyError:
            temp = 0
            
        list_temp.append(temp)
                
    texts_index.append(list_temp)
    
num_tokens = [ len(tokens) for tokens in data_test.tokenize]
num_tokens = np.array(num_tokens)
max_len = np.max(num_tokens)


# 生成三维的y
y = pd.Categorical(data_test.事故类型).codes
y_one_hot = np.zeros((y.shape[0], 3), dtype=np.int32)

for i in range(y.shape[0]):
    if y[i] == 0:
        y_one_hot[i, :] = [1.0,0.0,0.0]
    if y[i] == 1:
        y_one_hot[i, :] = [0.0,1.0,0.0]
    if y[i] == 2:
        y_one_hot[i, :] = [0.0,0.0,1.0]
        
embedding_matrix = []
for i in range(512): # 选用index排在512之前的词汇
    temp = model[model.wv.index2word[i]]
    embedding_matrix.append(temp)

embedding_matrix = np.array(embedding_matrix)
embedding_matrix = embedding_matrix.astype('float32') # 生成embedding matrix

texts_index = pad_sequences(texts_index, maxlen=max_len,
                            padding='pre', truncating='pre') # 对训练集进行pad和truncate
texts_index[texts_index >= 512] = 0

#sample_list = []
#for i in np.unique(y):
#    sample_index = np.random.choice(np.where(y==i)[0],90)
#    sample_list.extend(sample_index)
#y_select = y_one_hot[sample_list]
#X_select = texts_index[sample_list]  # 生成X 和 Y

X_train, X_test, y_train, y_test = train_test_split(texts_index,y_one_hot,test_size = 0.2, random_state = 0)

def lstm(): # 创建lstm神经网络模型
    model = Sequential()
    model.add(Embedding(512,
                        100,
                        weights=[embedding_matrix],
                        input_length=29,
                        trainable=False))

    model.add(LSTM(output_dim=50, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
  
   # model.summary()

    return model

lstm_model = lstm()
lstm_model.compile(loss = 'categorical_crossentropy',
         optimizer=Adam(lr=0.0001, decay=1e-6),
         metrics=['accuracy'])
lstm_model.fit(X_train,y_train,
         batch_size=32,
         shuffle = True,
         epochs=200,
         validation_data=(X_test, y_test),)


# 比较预测效果
y_pred = lstm_model.predict(texts_index)
y_cls = []
y_pred_cls = []
for i in range(400):
   
    y_pred_max = max(y_pred[i])
    y_max = max(y_one_hot[i])

    
    y_pred_index = np.where(y_pred[i]==y_pred_max)
    
    y_index = np.where(y_one_hot[i]==y_max)
    
    
    y_pred_cls.append(int(y_pred_index[0]))
    y_cls.append(int(y_index[0])) 
    
print("Confusion Matrix...")
cm = confusion_matrix(y_cls, y_pred_cls)
print(cm)

model_json = lstm_model.to_json()
with open("lstm_model.json", "w") as json_file:
    json_file.write(model_json)
lstm_model.save_weights("lstm_model.h5")