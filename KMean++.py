# -*- coding: utf-8 -*-
import numpy
import math
from random import choice
from copy import copy
from time import time
import pickle

import re
r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

FLOAT_MAX = 1e100

from dA import test_x_y_dA

class Point:
    __slots__ = ["Sent", "group","SentVec","number"]
    def __init__(self, Sent=None, group=0,number=0,SentVec=None):
        self.Sent, self.group ,self.SentVec,self.number= Sent,group,SentVec,number

#通过RAD将每个句子的词向量计算出来
def points_Vec_Rad(file_sent):
    MyVec={}
    MyNumber={}
    #将存储的词向量模型取出
    pkl_file=open('data/user_vec.pkl', 'r')
    data=pickle.load(pkl_file)
    pkl_file.close()
    #read sentence
    f = open(file_sent,'r')
    for mysent in f:
        line=re.sub(r,"",mysent.split('. ')[0])
        # print line
        word_Seq=line.split(' ')
        # print(word_Seq)
        pipei_y=list(data[word_Seq[0]])#first word
        for word in word_Seq:
            wordVec_Two_Merge=numpy.array(list(pipei_y)+list(data[word]))#合并两个词向量
            pipei_y=test_x_y_dA(wordVec_Two_Merge)
        MyVec[line]=pipei_y
        #构造句子和句子number的字典
        number=mysent.split('. ')[1].strip('\n')
        MyNumber[line]=number
        # print number
    f.close()
    #存储句子和向量的字典数据
    fout=open(filename,'w')
    pickle.dump(MyVec,fout)
    pickle.dump(MyNumber,fout)
    fout.close()

#不做任何处理将每个句子的词向量计算出来
def points_Vec_Add(file_sent):
    MyNumber={}
    MyVec={}
    #将存储的词向量模型取出
    pkl_file=open('data/user_vec.pkl', 'r')
    data=pickle.load(pkl_file)
    pkl_file.close()
    #读取句子
    f = open(file_sent,'r')
    for mysent in f:
        #计算句子的向量
        line=re.sub(r,"",mysent.split('. ')[0])
        # print line
        word_Seq=line.split(' ')
        # print(word_Seq)
        SUM_VEC=numpy.array(data[word_Seq[0]])
        for word in word_Seq:
            SUM_VEC =numpy.array(SUM_VEC+numpy.array(data[word]))#add two word vector  有待提高
        MyVec[line]=SUM_VEC
        # print SUM_VEC
        #构造句子和句子number的字典
        number=mysent.split('. ')[1].strip('\n')
        MyNumber[line]=number
        # print number
    f.close()
    #存储句子和向量的字典数据
    fout=open(filename,'w')
    pickle.dump(MyVec,fout)
    pickle.dump(MyNumber,fout)
    fout.close()

def generate_points():
    """将文件里面的句子读入point集里面"""
    pkl_file=open(filename,'r')
    data = pickle.load(pkl_file)
    labelNumber=pickle.load(pkl_file)
    pkl_file.close()
    MySent=data.keys()
    points = [Point() for _ in xrange(len(data))]
    for i in xrange(len(data)):
        points[i].Sent = MySent[i]
        points[i].SentVec =data[MySent[i]]
        points[i].number = labelNumber[MySent[i]]
    return points

def similarity(w1, w2):
    """计算词向量的相似度，使用余弦"""
    vec = numpy.dot(w1, w2)
    veclen=numpy.sqrt(numpy.sum(numpy.array(w1)**2))*numpy.sqrt(numpy.sum(numpy.array(w2)**2))
    if veclen>0.0:
        return vec / veclen
    else:
        print 'error vec'
        return ('error vec')

def nearest_cluster_center(point, cluster_centers):
    """计算每个点到聚类中心的距离，返回每个点的最近的聚类中心和距离"""
    min_index = point.group
    min_dist = FLOAT_MAX
    for i, cc in enumerate(cluster_centers):
        if cc.Sent == None:
            continue
        else:
            d = 1-similarity(cc.SentVec, point.SentVec)
            if min_dist > d:
                min_dist = d
                min_index = i
    return (min_index, min_dist)

#选择批次距离尽可能远的K个点
def K_findseed(points, cluster_centers):
    """初始化聚类中心，并将点分到这些聚类中心"""
    cluster_centers[0] = copy(choice(points))
    d = [0.0 for _ in xrange(len(points))]
    for i in xrange(1, len(cluster_centers)):
        for j, p in enumerate(points):
            d[j] = nearest_cluster_center(p, cluster_centers[:i])[1]
        max=d[0];index_max=0
        for j, di in enumerate(d):
            if(di>max):
                max=di
                index_max=j
            else:
                continue
        cluster_centers[i] = copy(points[index_max])
        cluster_centers[i].group=i
    for p in points:
        p.group = nearest_cluster_center(p, cluster_centers)[0]

#initialize K_centre_point
def K_findseed_1(points, cluster_centers):
    """初始化聚类中心，并将点分到这些聚类中心"""
    for i in xrange(0, len(cluster_centers)):
        cluster_centers[i] = copy(choice(points))
        cluster_centers[i].group=i
        print cluster_centers[i].Sent
    for p in points:
        p.group = nearest_cluster_center(p, cluster_centers)[0]

def comp_cluster_centers(Cluster_centre):
    """计算聚类中心的向量"""
    Cluster_centre_matrix=numpy.mat(Cluster_centre)
    L=numpy.sum(Cluster_centre_matrix,axis=0)
    M=L[0]*(1.0/len(Cluster_centre))
    return numpy.array(M)[0]

def K_mean_plus_mean(points , cluster_centers):
    """初始化聚类中心，并将点分到这些聚类中心"""
    changed = 0
    wordLists=[list() for _ in xrange(len(cluster_centers))]
    while True:
        for p in points:
            p.group = nearest_cluster_center(p, cluster_centers)[0]
            wordLists[p.group].append(p.SentVec)
        for i,cc in enumerate(cluster_centers):
            cc.SentVec=comp_cluster_centers(wordLists[i])
        changed = 0
        for p in points:
            min_i = nearest_cluster_center(p, cluster_centers)[0]
            if min_i != p.group:
                changed += 1
                p.group = min_i
        if changed ==0:
            break
    for i, cc in enumerate(cluster_centers):
        cc.group = i

def SSE_Points(points,cluster_centers):
    Sum_dis=0.0
    for p in points:
        Distance=(1-similarity(p.SentVec, cluster_centers[p.group].SentVec))**2
        # print Distance
        Sum_dis=Sum_dis+Distance
    return Sum_dis

def test(k=7):
    #initialize points information
    points = generate_points()
    #initialize K_cluster_centre
    cluster_centers = [Point() for _ in xrange(k)]
    K_findseed(points, cluster_centers)
    #compute K_mean
    K_mean_plus_mean(points,cluster_centers)
    #compute SSE(sum of the squared errors)
    SSE_sum=SSE_Points(points, cluster_centers)
    # print("%s : %.5f" % ("SSE", SSE_sum))
    return SSE_sum,points,cluster_centers

def test_show(k):
    t0=time()
    Min_SSE=numpy.inf
    for i in range(20):
        sse,ps,cs=test(k)
        if sse < Min_SSE:
            Min_SSE=sse
            best_Cluster=ps
            best_Cluster_centre=cs
    print "best cluster result:"
    print Min_SSE
    Group_list=[list() for _ in xrange(k)]
    for i in range(len(best_Cluster_centre)):
        # print best_Cluster_centre[i].Sent,best_Cluster_centre[i].group
        print u"-------这里输出第",i,u"类文本的句子------"
        for p in best_Cluster:
            if(int(p.number)==i):
                print p.Sent,p.group,p.number
                Group_list[i].append(str(p.group))
    print ("%s :%.5fS"%("spend times",time()-t0))
    return best_Cluster,Group_list

#compute entropy
def cacShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for currentLabel in dataset:
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*math.log(prob, 2)
    return shannonEnt
def test_entropy(best_Cluster,k,gl):
    sum_entropy=0.0
    for i in range(k):
        print "entropy:",cacShannonEnt(gl[i])
        sum_entropy=sum_entropy+((len(gl[i])*1.0)/len(best_Cluster))*cacShannonEnt(gl[i])
    print sum_entropy

#compute purity
def cacpurityEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for currentLabel in dataset:
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    purityEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        if prob > purityEnt:
            purityEnt = prob
    return purityEnt
def test_purity(best_Cluster,k,gl):
    sum_purity=0.0
    for i in range(k):
        print "purity:",cacpurityEnt(gl[i])
        sum_purity=sum_purity+((len(gl[i])*1.0)/len(best_Cluster))*cacpurityEnt(gl[i])
    print sum_purity

#当test.txt改变执行
# filename="data/MyVec_Rad.pkl"
# points_Vec_Rad('data/test_10.txt')
filename="data/MyVec_add.pkl"
points_Vec_Add('data/test_10.txt')
bps,gl = test_show(k=10)
test_entropy(bps,10,gl)
test_purity(bps,10,gl)
