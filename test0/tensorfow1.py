#coding=utf-8
import numpy as np
import tensorflow as tf
import random
import _compat_pickle
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
pos_file='pos.txt'
neg_file='neg.txt'
def create_lexicon(pos_file,neg_file):
    lex=[]
    def process_file(f):#读取文件进行分词完成后返回词汇数组
        with open(pos_file,'rb') as f:
            lex=[]
            lines = f.readline()
            for line in lines:
                words = nltk.word_tokenize(line.lower())#把所有大写字母转换为小写然后进行分词
                lex+=words
            return lex

    lex+=process_file(pos_file)
    lex+=process_file(neg_file)
    lemmatizer =WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word)for word in lex]#将词汇还原为一般形式列入cars -> car
    word_count = Counter(lex)#统计词汇频率
    #print(word_count)
    #{".",1354,"the",1564.....}
    lex=[]
    for word in word_count:
        if word_count[word]<2000 and word_count[word] > 20:#去掉常用词和不常用词
            lex.append(word)
    return  lex

lex = create_lexicon(pos_file,neg_file)

#lex里保存了文本中出现的单词

#把每条评论转换为向量，假设lex为[man ,hello,word,great,bad....]
#评论 i think this movie is great转换为[0,0,0,0,0,1]将出现过的词汇记为1没有出现过的记为0

def normalize_dataset(lex):
    dataset = []

    #lex :词汇表;review:评论;clf:评论对应的类型;
    def string_to_vector(lex,review,clf):
        pass


