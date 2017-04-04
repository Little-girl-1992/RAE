# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy 
import pprint, pickle
def train_model():
	"""训练一个词向量模型"""
	model = Word2Vec(LineSentence('temp/data.txt'), size=300, window=5, min_count=1, workers=4)
	model.save('temp/temp.bin')
	model.save_word2vec_format('temp/temp.txt', binary=False)

# train_model() #执行这个函数

def used_model_m():
	"""测试一个词的词向量"""
	model =Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',binary=True)
	# model =Word2Vec.load('temp/temp.bin')
	b=model['spilt']
	print b

# used_model_m() #执行这个函数


def used_model():
	"""使用Word2Vec模块自带的函数，计算两个句子之间的相似度"""
	model =Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',binary=True)
	b=model.similarity(u'the',u'to')
	print (b)
	# return (b)

# used_model() #执行这个函数


def used_model():
	"""使用Word2Vec模块自带的函数，计算两个句子之间的相似度"""
	model =Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',binary=True)
	fouts =open('data/model_Dict.txt', 'w')#以写得方式打开文件
	for word in model.vocab:
		fouts.write(word)
	fouts.close()
	# return (b)
# used_model() #执行这个函数

def read_model():
	"""将语料里面的词的词向量都读出来，另存在data/user_vec.pkl文件里面，方便读取"""
	model =Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',binary=True)

	outVec=dict()
	fouts =open('data/user_Dict_Error.txt', 'w')#以写得方式打开文件
	with open('data/user_Dict.txt', 'r') as fin:#以读的方式打开文件
		for line in fin:
			s=line.split(' ')[0]
			if s in model.vocab:
				b=model[s]
				outVec[s]=b
				print s
			else:
				print "这个单词有问题",s
				fouts.write(s+'\n')
				outVec[s]=[0]*300
	fout= open('data/user_vec.pkl', 'w')#以写得方式打开文件
	pickle.dump(outVec,fout)
	fout.close()
	fin.close()
	fouts.close()
# read_model() #执行这个函数

def read_model():
	"""将语料里面的词的词向量都读出来，另存在data/user_vec.pkl文件里面，方便读取"""
	outVec=[]
	pkl_file=open('data/user_vec.pkl', 'r')
	data=pickle.load(pkl_file)
	with open('data/user_Dict.txt', 'r') as fin:#以读的方式打开文件
		for line in fin:
			s=line.split(' ')[:-1][0]
			vec=data[s]
			outVec.append(vec)
	fout= open('data/user_vec_D.pkl', 'w')#以写得方式打开文件
	pickle.dump(numpy.array(outVec),fout)
	pkl_file.close()
	fout.close()
	fin.close()
	#用于显示数据
	print('read pkl...')

# read_model() #执行这个函数


import re
r='[’ !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'

def merge_two():
	"""将文件里面句子的词向量两个两个的合并在一起
		e.g 123456  合并成12 23 34 45 56"""
	pkl_file=open('data/user_vec.pkl', 'r')
	data=pickle.load(pkl_file)

	datafile=open('data/English_data.txt','r')
	fout= open('data/user_merge.pkl', 'w')#以写得方式打开文件
	for dataitem in datafile:
		#去除句子里面的标点符号
		item_remove=re.sub(r," ",dataitem)
		dataSent=item_remove.split(' ')[:-1]
		print dataSent
		for i in range(0,len(dataSent)-1):
			c=list(data[dataSent[i]])+list(data[dataSent[i+1]])
			pickle.dump(numpy.array(c),fout)
	fout.close()
# merge_two()  #执行这个函数

def merge_sent():
	"""将文件里面句子的词向量合并在一起"""
	pkl_file=open('data/user_vec.pkl', 'r')
	data=pickle.load(pkl_file)

	datafile=open('data/English_data.txt','r')
	fout= open('data/user_merge_sent.pkl', 'w')#以写得方式打开文件
	for dataitem in datafile:
		#去除句子里面的标点符号
		item_remove=re.sub(r," ",dataitem)
		dataSent=item_remove.split(' ')[:-1]
		print dataSent
		c=list()
		for word in dataSent:
			c.append(list(data[word]))
		pickle.dump(numpy.array(c,dtype=numpy.float32),fout)
	fout.close()

# merge_sent()  #执行这个函数


def split_sent():
	"""将文件里面的一部分抽取出来"""
	outVec=[]
	pkl_file=open('data/user_merge_sent.pkl', 'r')
	for i in range(100):
		data=pickle.load(pkl_file)
		outVec.append(data)
	pkl_file.close()
	fout= open('data/user_merge_sent_100.pkl', 'w')#以写得方式打开文件
	pickle.dump(numpy.array(outVec),fout)
	fout.close()
	#用于显示数据
	print('read pkl...')

split_sent() #执行这个函数

