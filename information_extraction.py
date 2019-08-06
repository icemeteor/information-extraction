import pandas as pd
import numpy as np
import jieba
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

class IE(object):
    
    def __init__(self):
        self.x_train_shape = 0
        self.num_classes = 0
    
    def preprocess(self,data,features,tags,n=90): # 对于Naive Bayes和SVM模型的数据预处理函数
        # data 为选用的数据集, features为选用的特征列名称, tags为预测值列名称, n为提取数据行数.
        data_use = pd.DataFrame()
        for i in data[tags].unique():
            row = data[data[tags] == i].sample(n=n)
            data_use = pd.concat([data_use,row],ignore_index=True)
        
        data_use = data_use[[features,tags]] 
        data_use[features] = data_use[features].str.replace('[^\w]','')
        data_use['tokenize'] = data_use[features].apply(jieba.lcut)
        data_use.tokenize = data_use.tokenize.apply(' '.join)
        
        y = data_use[tags]
        X = data_use.tokenize
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
        
        return X_train, X_test, y_train, y_test #返回训练集和测试集
        
    def preprocess_NN(self, data, features, tags, n=90): # 对于神经网络模型的数据预处理函数
        # data 为选用的数据集, features为选用的特征列名称, tags为预测值列名称, n为提取数据数量数.
        data_use = pd.DataFrame()
        for i in data[tags].unique():
            row = data[data[tags] == i].sample(n=n)
            data_use = pd.concat([data_use,row],ignore_index=True)
        
        data_use = data_use[[features,tags]] 
        data_use[features] = data_use[features].str.replace('[^\w]','')
        data_use['tokenize'] = data_use[features].apply(jieba.lcut)
        data_use.tokenize = data_use.tokenize.apply(' '.join)
        
        y = data_use[tags]
        X = data_use.tokenize
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

        train_posts = X_train
        train_tags = y_train
        test_posts = X_test
        test_tags = y_test

        tokenize = text.Tokenizer(char_level=False) # 功能类似CountVectorizer和TFIDF
        tokenize.fit_on_texts(train_posts) # 只用训练集拟合

        x_train = tokenize.texts_to_matrix(train_posts) #生成训练集matrix
        x_test = tokenize.texts_to_matrix(test_posts) # 生成测试集matrix

        encoder = LabelEncoder()
        encoder.fit(train_tags)
        y_train = encoder.transform(train_tags)
        y_test = encoder.transform(test_tags)
    
        self.num_classes = np.max(y_train) + 1
        y_train = utils.to_categorical(y_train, self.num_classes) # 将文字类型数据转换为分类数据
        y_test = utils.to_categorical(y_test, self.num_classes)
        
        self.x_train_shape = x_train.shape[1]
        
        return x_train, x_test, y_train, y_test
        
    def NB(self): # Navie Bayes 模型
        nb = Pipeline([('tfidf',TfidfVectorizer()),('clf',MultinomialNB())])
        return nb
    
    def SVM(self): # Support Vector Machine 模型
        sgd = Pipeline([('tfidf', TfidfVectorizer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=10, tol=None)),])
        return sgd
    
    def NN(self): # 基础神经网络模型
        model = Sequential()
        model.add(Dense(512, input_shape=(self.x_train_shape,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model
    