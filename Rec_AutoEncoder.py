# -*- coding: utf-8 -*-

from time import time

import numpy
import pickle
import random

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class RAE(object):
    """这里定义了一个类，这个类包含了Denoising Auto-Encoder所用的数据、函数"""
    #初始化数据，input是输入数据，n_visible是输入输出的向量空间维度，
    # n_hidden是隐藏层的向量空间维度，W,bhid,bvis是神经网络参数
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=600,
        n_hidden=300,
        W1=None,
        W2=None,
        bhid=None,
        bvis=None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.W =theano.shared(value=W1, name='W1', borrow=True) 
        self.W_prime = theano.shared(value=W2, name='W2', borrow=True)
        self.b = theano.shared(value=bhid, name='bhid', borrow=True)
        self.b_prime = theano.shared(value=bvis, name='bvis', borrow=True)
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.W_prime, self.b, self.b_prime]

    def get_input_values(self,origin):
        """ Computes the values of the input layer """
        c1=origin[:-1];c2=origin[1:]
        return T.concatenate([c1,c2],axis=1)

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return (T.tanh(T.dot(input, self.W) + self.b).T/T.sum((T.tanh(T.dot(input, self.W) + self.b)).T**2,axis=0)**0.5).T

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer
        """
        return T.tanh(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_min_index(self,origin):
        """ This function computes the cost and the updates for one trainng
        step of the RAE """
        merge_input = self.get_input_values(origin)
        encode = self.get_hidden_values(merge_input)
        decode = self.get_reconstructed_input(encode)

        L = T.sum((0.5*numpy.array(decode-merge_input)**2), axis=1)
        min_index=T.argmin(L)
        return (min_index,merge_input[min_index],encode[min_index])

    def get_cost_updates(self,learning_rate,merge_input):
        """ This function computes the cost and the updates for one trainng
        step of the RAE """

        encode = self.get_hidden_values(merge_input)
        decode = self.get_reconstructed_input(encode)
        # computes the cost by ordinary least squares(OLS)
        L = T.sum((0.5*numpy.array(decode-merge_input)**2), axis=1)
        cost = T.mean(L)
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return (cost, updates)



def train_RAE(learning_rate=0.1, training_epochs=500,n_train_Sent=100,
            dataset='../data/user_merge_sent_10000.pkl'):
    """训练一个600-300-600的神经网络 """

    #读取输入数据
    print('read pkl...')
    pkl_file=open(dataset, 'r')
    train_set_x=pickle.load(pkl_file)
    pkl_file.close()

    W,W_prime,b,b_p=read_W_b()

    #定义函数的符号
    x = T.matrix('x')
    origin = T.matrix('origin')
    words = T.matrix('words')
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    rae = RAE(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        W1=W,
        W2=W_prime,
        bhid=b,
        bvis=b_p,
        n_visible=600,
        n_hidden=300
    )
    min_index=rae.get_min_index(origin=origin)
    get_y=theano.function(
        inputs=[origin],
        outputs=min_index)
    cost, updates = rae.get_cost_updates(
        learning_rate=learning_rate,
        merge_input=words
    )
    train_rae = theano.function(
        inputs=[words],
        outputs=cost,
        updates=updates
    )
    #开始的时间
    start_time = time()

    ############
    # TRAINING #
    ############

    #将数据分批带入函数符号里面，开始训练模型，记录cost，更新参数
    for epoch in range(training_epochs):
        words_input=[]
        c = []
        for batch_index in range(n_train_Sent):
            """将句子里面的词按还原度最低的合并"""
            reinput_list=train_set_x[batch_index]
            while (len(reinput_list)>1):
                min_index,min_x,min_y=get_y(numpy.matrix(reinput_list))
                words_input.append(min_x)

                sentList_buff=[]
                for i in range(len(reinput_list)):
                    if i < min_index:
                        sentList_buff.append(reinput_list[i])
                    elif i == min_index:
                        sentList_buff.append(min_y)
                    elif i > (min_index+1):
                        sentList_buff.append(reinput_list[i])
                    else:
                        continue
                reinput_list=sentList_buff
        cost=train_rae(numpy.matrix(words_input))
        print('Training epoch %d, cost: %f' % (epoch, numpy.mean(cost)))

    #将w,b记录在文件里面。
    fout= open('test1data/user_Wb_t2.pkl', 'w')#以写得方式打开文件
    pickle.dump(rae.W.get_value(borrow=True).T,fout)
    pickle.dump(rae.W_prime.get_value(borrow=True).T,fout)
    pickle.dump(rae.b.get_value(borrow=True).T,fout)
    pickle.dump(rae.b_prime.get_value(borrow=True).T,fout)
    fout.close()

    print('训练时间：%s'%(time() - start_time))

def read_W_b():
    """读取文件里面的参数W，b"""
    pkl_file = open('test1data/user_Wb_t1.pkl', 'r')
    W = pickle.load(pkl_file).T
    # print W
    W_prime = pickle.load(pkl_file).T
    # print W_prime
    b = pickle.load(pkl_file).T
    b_p = pickle.load(pkl_file).T
    pkl_file.close()
    return W,W_prime,b,b_p

#训练网络，生成参数
# train_RAE()


