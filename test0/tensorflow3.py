import tensorflow as tf
import numpy as np

#下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/',one_hot=True)

#一张图片是28*28 RNN将其分成块
chunk_size = 28 #块大小
chunk_n = 28

run_size = 256

n_output_layer = 10 #输出层

X = tf.placeholder('float',[None,chunk_n,chunk_size])
Y = tf.placeholder('float')

#定义带训练的神经网络
def recurrent_neural_network(data):
    layer = {'w_':tf.Variable(tf.random_normal([run_size,n_output_layer])),'b_':tf.Variable(tf.random_normal([n_output_layer]))}
    #生成符合正太分布的随机值
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(run_size)#定义BasicLSTMCell

    data = tf.transpose(data,[1,0,2])#转置
    data = tf.reshape(data,[-1,chunk_size])#将矩阵转换为28列
    data = tf.split(0,chunk_n,data)#将矩阵切割
    outputs,status = tf.nn.rnn(lstm_cell,data,dtype = tf.float32)

    output = tf.add(tf.matmul(outputs[-1],layer['w_']),layer['b_'])#矩阵相乘再加上b_

    return output

batch_size = 100

def train_neural_network(X,Y):
    predict = recurrent_neural_network(X)

    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict,Y))
    #计算loss
    optimizer = tf.train.AdamOptimizer.minimize(cost_func)
    #优化cost_func

    epochs = 13

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        epoch_loss = 0
        for epoch in range(epochs):
            for i in range(int(mnist.train.num_examples/batch_size)):
                x,y = mnist.train.next_batch(batch_size)
                x = x.reshape([batch_size,chunk_n,chunk_size])
                _,c = session.run([optimizer,cost_func],feed_dict={X:x,Y:y})
                epoch_loss += c

            print(epoch,':',epoch_loss)
        correct = tf.equal(tf.argmax(predict,1),tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))

        print('准确率：',accuracy.eval({X:mnist.test.images.reshape(-1,chunk_n,chunk_size),Y:mnist.test.labels}))
train_neural_network(X,Y)
