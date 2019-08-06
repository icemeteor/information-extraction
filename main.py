# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import jieba
import re
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.models import model_from_json
import warnings

warnings.simplefilter('ignore')


def car_logo(data): # 查出车的品牌
    data_car = pd.read_excel('20180803_car_logo.xls',sheet_name = 0)
    logo = np.array(data_car['品牌'].dropna())
    np.append(logo,'公交车')
    
    logo_tmp = []
    desc = data['警情描述']
    for i in range(len(desc)):
        desc_logo = []
        word = jieba.lcut(desc.iloc[i])
        for j in range(len(word)):
            if word[j] in logo:
                desc_logo.append(word[j])
        logo_tmp.append(desc_logo)
    return logo_tmp

def car_number(data): # 查出车的车牌
    province = ['京','津','冀','晋','蒙','辽','吉','黑','沪','苏','浙','皖','闽','赣','鲁',
           '豫','鄂','湘','粤','桂','琼','川','贵','滇','渝','藏','陕','甘','青','宁',
           '新','港','澳','台']
    plate_number = []
    desc = data['警情描述']
    for i in range(len(desc)):
        desc_number = []
        tmp = desc.iloc[i].replace(' ','')
        word = jieba.lcut(tmp)
        for j in range(len(word)):
            if len(word[j]) == 6:
                #plate_number.append(word[j-1][-1] + word[j].upper())
                if word[j-1][-1] not in province:
                    desc_number.append('鲁' + word[j].upper())
                else:
                    desc_number.append(word[j-1][-1] + word[j].upper())
        plate_number.append(desc_number)
    return plate_number

def choose_token(token): # 删除word2vec模型中不存在的词语和字符
    model = Word2Vec.load('word2vec.model') 
    
    voc_list = list(model.wv.vocab.keys())
    token_tmp = [value for value in token if value in voc_list]
    return token_tmp

def accident_type(data): # 提出事故类型
    # 做对于输入数据的预处理
    data_use = data
    data_use['警情描述'] = data_use['警情描述'].str.replace('[^\w]','')
    data_use['tokenize'] = data_use['警情描述'].apply(jieba.lcut)
    data_use.tokenize = data_use.tokenize.apply(' '.join)
    data_use.tokenize = data_use.tokenize.str.split(' ')
    
    data_use.tokenize = data_use.tokenize.apply(choose_token)
    
    # 载入 Word2Vec model 
    
    model = Word2Vec.load('word2vec.model') 
    
    # 载入 lstm model
    json_file = open('lstm_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    lstm_model = model_from_json(loaded_model_json)
    # 载入 weights into new model
    lstm_model.load_weights("lstm_model.h5")
    
    texts_index = [] 

    for i, text in enumerate(data_use.tokenize):
        list_temp = []
        for t, word in enumerate(text):
            try:
                temp = model.wv.vocab[word].index
            
            except KeyError:
                temp = 0
            
            list_temp.append(temp)
                
        texts_index.append(list_temp)
    # 记录所有字符的index
    
    texts_index = pad_sequences(texts_index, maxlen=29,
                            padding='pre', truncating='pre') 
    texts_index[texts_index >= 512] = 0 # 选用前512个index
    
    lstm_model.compile(loss = 'categorical_crossentropy',
         optimizer=Adam(lr=0.0001, decay=1e-6),
         metrics=['accuracy'])
    
    y_pred = lstm_model.predict(texts_index)
    
    y_pred_cls = []
    for i in range(data_use.shape[0]):
        y_pred_max = max(y_pred[i])
        y_pred_index = np.where(y_pred[i]==y_pred_max)
        y_pred_cls.append(int(y_pred_index[0]))
    
    y_res = []
    
    #将生成的分类转换成文字
    for i in range(len(y_pred_cls)):
        if y_pred_cls[i] == 0:
            y_res.append('刮擦')
        elif y_pred_cls[i] == 1:
            y_res.append('未知')
        elif y_pred_cls[i] == 2:    
            y_res.append('相撞')
    
    
    return y_res

def report(data): # 生成结果表格
    location = data['警情地点']
    logo = car_logo(data)
    number = car_number(data)
    accident = accident_type(data)
    
    res = pd.DataFrame({'desc':data['警情描述'],
                       'location':location,
                       'logo':logo,
                       'number':number,
                       'accident':accident})
    return res

if __name__ == '__main__':
    # 实验数据
    data = pd.read_excel("police_data.xls",sheet_name=0)
    data_use = data[500:1000]
    report_performace = report(data_use)
    report_performace.to_csv('report.csv',encoding = 'utf-8-sig')
    print(report_performace.head(20))