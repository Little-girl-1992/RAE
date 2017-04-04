# -*- coding: utf-8 -*-

from __future__ import print_function

import timeit

import numpy
import pickle

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
# from gensim.models import Word2Vec

class dA(object):
    """这里定义了一个类，这个类包含了Denoising Auto-Encoder所用的数据、函数"""
    #初始化数据，input是输入数据，n_visible是输入输出的向量空间维度，
    # n_hidden是隐藏层的向量空间维度，W,bhid,bvis是神经网络参数
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=300,
        n_hidden=150,
        W=None,
        bhid=None,
        bvis=None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if W==None:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            print('这里W是空的')
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if bvis==None:
            print('这里bvis是空的')
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='bvis',
                borrow=True
            )

        if bhid==None:
            print('这里bhid是空的')
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='bhid',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level"""
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.tanh(T.dot(input, self.W) + self.b)


    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer
        """
        return T.tanh(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # computes the cost by ordinary least squares(OLS)
        L = T.sum((numpy.array(self.x -z)**2), axis=1)
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)


def test_dA(learning_rate=0.05, training_epochs=50,
            dataVec='data/user_vec_D.pkl',
            batch_size=20):
    """训练一个300-150-300的神经网络"""

    #读取输入数据
    print('read pkl...')
    Vec_file=open(dataVec, 'r')
    train_set_x=pickle.load(Vec_file)
    # print (train_set_x)

    # 计算每批的数量
    n_train_batches = numpy.matrix(train_set_x).shape[0] / batch_size

    #定义函数的符号
    x = T.matrix('x') 
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=300,
        n_hidden=150
    )
    cost, updates = da.get_cost_updates(
        corruption_level=0.0,
        learning_rate=learning_rate
    )
    train_da = theano.function(
        inputs=[x],
        outputs=cost,
        updates=updates
    )
    #开始的时间
    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    cost_mean=[]
    for epoch in range(training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(train_set_x[batch_index*batch_size: (batch_index+1)* batch_size]))#将数据分批带入函数符号里面，开始训练模型，记录cost
        print('Training epoch %d, cost ' % epoch, numpy.mean(c))
        cost_mean.append(numpy.mean(c))


    f= open('data/user_cost_data_0.0.pkl', 'w')
    pickle.dump(cost_mean,f)
    f.close()
    #将w,b记录在data/user_Wb_data_0.0.pkl文件里面。
    fout= open('data/user_Wb_data_0.0.pkl', 'w')#以写得方式打开文件
    pickle.dump(da.W.get_value(borrow=True).T,fout)
    pickle.dump(da.b.get_value(borrow=True).T,fout)
    pickle.dump(da.b_prime.get_value(borrow=True).T,fout)
    fout.close()

    print('W:',numpy.mat(da.W.get_value(borrow=True).T))
    print('bhid:',numpy.mat(da.b.get_value(borrow=True).T))
    print('bvis:',numpy.mat(da.b_prime.get_value(borrow=True).T))

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    print('训练时间：%s'%(training_time)) 

def test_x_z_dA(x_x):
    """将输入数据样本带入，计算重构的Z"""
    #读入训练好的参数
    W,b,b_p=read_W_b()
    #定义函数符号
    x = T.matrix('x')
    y = T.matrix('y')
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        W=W.T,
        bhid=b.T,
        bvis=b_p.T
    )
    y = da.get_hidden_values([x])
    z = da.get_reconstructed_input([y])
    train_da = theano.function(
        inputs=[x],
        outputs=z
    )
    #开始计算重构的Z
    x_z=train_da(numpy.mat(x_x))
    return (x_z[0,0,0])

def test_x_y_dA(x_x):
    """将输入数据样本带入，计算隐藏层Y"""
    #读入训练好的参数
    W,b,b_p=read_W_b()
    #定义函数符号
    x = T.matrix('x')
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        W=W.T,
        bhid=b.T,
        bvis=b_p.T
    )
    y = da.get_hidden_values([x])
    train_da = theano.function(
        inputs=[x],
        outputs=y
    )
    #开始计算隐藏层Y
    x_y=train_da(numpy.mat(x_x))
    return (x_y[0,0])


def read_W_b():
    """读取文件里面的参数W，b"""
    pkl_file = open('data/user_Wb_data_0.0.pkl', 'r')
    data1 = pickle.load(pkl_file)
    # pprint.pprint(data1)
    data2 = pickle.load(pkl_file)
    # pprint.pprint(data2)
    data3 = pickle.load(pkl_file)
    # pprint.pprint(data3)
    pkl_file.close()
    return data1,data2,data3

def similarity(w1, w2):
    """计算词向量的相似度，使用余弦"""
    vec = numpy.dot(w1, w2)
    veclen=numpy.sqrt(numpy.sum(numpy.array(w1)**2))*numpy.sqrt(numpy.sum(numpy.array(w2)**2))
    if veclen>0.0:
        return vec / veclen
    else:
        return ('error vec')

def used_Dict(word='A'):
    """返回词列表的词向量列表"""
    pkl_file=open('data/user_vec.pkl', 'r')
    data=pickle.load(pkl_file)
    b=data[word]
    return (b)


def picture_cost():
    """画出训练模型的cost的收敛过程"""
    import matplotlib.pyplot as plt
    pkl_file = open('199801/user_cost_199801_0.0.pkl', 'r')
    data1 = pickle.load(pkl_file)
    x=[i for i in range(1,101)]
    print(x)
    y=data1
    fig,ax = plt.subplots()
    plt.xlabel("times")
    plt.ylabel("cost")
    ax.plot(x,y)
    plt.show()

# picture_cost() #执行这个函数

def test_word():
    pass
    #训练网络，生成参数
    # test_dA()
    # x_x1=list(used_Dict(word='types'))
    # x_x2=list(used_model(u'比较'))

    #计算生成x_y x_z
    # x_z1=test_x_z_dA(x_x1)
    # x_z2=test_x_z_dA(x_x2)
    # x_y1=test_x_y_dA(x_x1)
    # x_y2=test_x_y_dA(x_x2)
    # x_y=test_x_y_dA(x_x)
    # print(x_z[0,0,0])
    # print(len(x_y[0,0]))


    #比较相似度
    # simly1=similarity(x_x1,x_x2)
    # simly2=similarity(x_z1,x_z2)
    # simly3=similarity(x_y1,x_y2)
    # simly=similarity(x_x1,x_z1)
    # simly=similarity((Nmama)*2,(Nmama)*2)
    # print(simly)
    # print(simly1)
    # print(simly2)
    # print(simly3)

test_word() #执行这个函数




    
