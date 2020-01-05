# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:53:56 2018
@author: zs
"""
import re
import time
import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# 导入数据集函数，返回聚类的数据与对应ID
def loadDataSet(filename):
    dataset = pd.read_csv(filename, encoding='utf-8')
    m, n = dataset.shape  # 获取行、列
    data = dataset.values[:, -1]
    dataID = dataset.values[:, 0]
    return data.reshape((m, 1)), dataID.reshape((m, 1))


# numpy 转化为 list
def ndarrayToList(dataArr):
    dataList = []
    m, n = dataArr.shape
    for i in range(m):
        for j in range(n):
            dataList.append(dataArr[i, j])
    return dataList


# 去掉字符串、特殊符号
def removeStr(listData):
    strData = "".join(listData)
    removeStrData = re.sub("[\s+\!\,$^*()+\"\']+:|[+——！，,《》“”〔【】；：。？、�./-~@#￥……&*（）]+", "", strData)
    return removeStrData


# 创建停用词列表
def stopwordslist(filePath):
    stopword = [line.strip() for line in open(filePath, 'r', encoding='utf-8').readlines()]
    return stopword


# 保存文件
def saveFile(filename):
    with open(filename, 'a') as fr:
        for line in dataSplit:
            strLine = ' '.join(line)
            fr.write(strLine)
            fr.write('\n')
        fr.close()


# 对数据集分词、去停用词
def wordSplit(data):
    stopword = stopwordslist('./data/stopwords.txt')  # 创建通用词列表
    word = ndarrayToList(data)
    m = len(word)
    wordList = []
    for i in range(m):
        rowListRemoveStr = removeStr(word[i])  # 去特殊符号
        rowList = [eachWord for eachWord in jieba.cut(rowListRemoveStr)]  # 分词
        removeStopwordList = []
        for eachword in rowList:
            if eachword not in stopword and eachword != '\t' and eachword != ' ':
                removeStopwordList.append(eachword)
        wordList.append(removeStopwordList)
    return wordList


# 计算 tf-idf 值
def TFIDF(wordList):
    corpus = []  # 保存预料
    for i in range(len(wordList)):
        wordList[i] = " ".join(wordList[i])
        corpus.append(wordList[i])
    # 将文本中的词语转换成词频矩阵,矩阵元素 a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    # 该类会统计每个词语tfidf权值
    transformer = TfidfTransformer()
    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重  
    weight = tfidf.toarray()

    return word, weight


# 对生成的 tfidf 矩阵做PCA降维
'''
权重矩阵非常稀疏，使用PCA降维(为什么不是SVD降维) SVD适合稠密矩阵降维
'''


def matrixPCA(weight, dimension):
    pca = PCA(n_components=dimension)  # 初始化PCA
    pcaMatrix = pca.fit_transform(weight)  # 返回降维后的数据
    print("降维之前的权重维度：", weight.shape)
    print("降维之后的权重维度：", pcaMatrix.shape)
    return pcaMatrix


# 层级聚类 birch  k-means适合维度低且速度慢
def birch(matrix, k):
    clusterer = Birch(n_clusters=k)  # 分成簇的个数
    y = clusterer.fit_predict(matrix)  # 聚类结果
    return y


# 计算轮廓系数
def Silhouette(matrix, y):
    silhouette_avg = silhouette_score(matrix, y)  # 平均轮廓系数
    sample_silhouette_values = silhouette_samples(matrix, y)  # 每个点的轮廓系数
    print(silhouette_avg)
    return silhouette_avg, sample_silhouette_values


# 画图
def Draw(silhouette_avg, sample_silhouette_values, y, k):
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(18, 7)
    # 第一个 subplot 放轮廓系数点
    # 范围是[-1, 1]
    ax1.set_xlim([-0.2, 0.5])
    # 后面的 (k + 1) * 10 是为了能更明确的展现这些点
    # ax1.set_ylim([0, len(X) + (k + 1) * 10])
    y_lower = 10

    for i in range(k):  # 分别遍历这几个聚类
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.spectral(float(i) / k)  # 搞一款颜色
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0,
                          ith_cluster_silhouette_values,
                          facecolor=color,
                          edgecolor=color,
                          alpha=0.7)
        # 在轮廓系数点这里加上聚类的类别号
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # 计算下一个点的 y_lower y轴位置
        y_lower = y_upper + 10
    # 在图里搞一条垂直的评论轮廓系数虚线
    ax1.axvline(x=silhouette_avg, color='red', linestyle="--")
    plt.show()


# 保存聚类结果
def saveResult(data, y):
    y = y.reshape((len(data), 1))
    for i in range(12):
        filename = './result1/result' + str(i) + '.csv'  # 文件名
        with open(filename, 'a', encoding='utf8') as fr:
            for j in range(13001):
                if y[j] == i:
                    strLine = ''.join(data[j])
                    fr.write(strLine)
                    fr.write('\n')
            fr.close()


if __name__ == "__main__":
    # start time
    start = time.clock()
    k = 12  # 聚成12类
    dataSplit=[]
    with open('result_chuli3.txt','r',encoding='gbk') as f:
        data = f.readlines()
        for i in range(len(data)):
            data[i]=data[i].strip()
            for j in data[i].split(' '):
                if j!=None:
                    dataSplit.append(j)
    word, weight = TFIDF(dataSplit)  # 生成 tfidf 矩阵
    weightPCA = weight

    # 将原始矩阵降维，降维后效果反而没有不降维的好
    # weightPCA = matrixPCA(weight, dimension = 1000)
    y = birch(weightPCA, k)
    silhouette_avg, sample_silhouette_values = Silhouette(weightPCA, y)  # 轮廓系数

    Draw(silhouette_avg, sample_silhouette_values, y, k)
    saveResult(data, y)  # 保存聚类结果，一类保存为一个csv文件

    elapsed = (time.clock() - start)
