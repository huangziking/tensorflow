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
    def string_to_vector(lex,review,clf):#对评论进行分类
        words = nltk.word_tokenize(line.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word)for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1
        return [features,clf]

    with open(pos_file,'r') as f:
        lines = f.readline()
        for line in lines:
            one_sample = string_to_vector(lex,line,[1,0])
            dataset.append(one_sample)
    with open(neg_file,'r') as f:
        lines = f.readline()
        for line in lines:
            one_sample = string_to_vector(lex,line,[0,1])
            dataset.append(one_sample)
    print(len(dataset))
    return dataset
dataset = normalize_dataset(lex)
random.shuffle(dataset)#对数据进行随机排列

#with open('save.cPickle','wb') as f:
     #cPickle.dump(dataset,f)将整理完成的数据进行保存
test_size = int(len(dataset)*0.1)
dataset = np.array(dataset)
train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]#取10%为测试数据其余90%为训练数据
#定义每层神经元个数
n_input_layer = len(lex) #输入层
n_layer_1 = 1000 #隐藏层1
n_layer_2 = 1000 #隐藏层2
n_output_layer=2 #输出层
#定义待训练的神经网络
def neural_network(data):
    #定义第一层的权重和biases
    layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer,n_layer_1])),
                   'b_':tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1,n_layer_2])),
                   'b_':tf.Variable(tf.random_normal([n_layer_2]))}
    layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2,n_output_layer])),
                        'b_':tf.Variable(tf.random_normal([n_output_layer]))}
    print(layer_1_w_b)
    print(layer_2_w_b)
    print(layer_output_w_b)
    layer_1 = tf.add(tf.matmul(data,layer_1_w_b['w_']),layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)#激活函数
    layer_2 = tf.add(tf.matmul(data,layer_2_w_b['w_']),layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.add(tf.matmul(layer_2,layer_output_w_b['w_']),layer_output_w_b['b_'])

    return layer_output


