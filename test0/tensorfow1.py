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
            lines = f.readlines()
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

batch_size = 50

X = tf.placeholder('float',[None,len(train_dataset[0][0])])
Y = tf.placeholder('float')

def train_neural_network(X,Y):
    predict = neural_network(X)
    cost_func = tf.reduce.mean(tf.nn.softmax_cross_entropy_with_logits(predict,Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    epochs = 13

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        epoch_loss = 0

        i=0
        random.shuffle(train_dataset)
        train_x = dataset[:,0]
        train_y = dataset[:,1]

        for epoch in range(epochs):
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]

                _,c = session.run([optimizer,cost_func],feed_dict={X:list(batch_x),Y:list(batch_y)})

                epoch_loss += c
                i += batch_size
            print(epoch,':',epoch_loss)

        text_x = test_dataset[:,0]
        text_y = test_dataset[:,1]
        correct = tf.equal(tf.argmax(predict,1),tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('准确率：',accuracy.eval({X:list(text_x),Y:list(text_y)}))
train_neural_network(X,Y)


