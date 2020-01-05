# -*- coding: UTF-8 -*-
__author__ = 'zy'
__time__ = '2019/4/9 16:50'
from time import time
from sklearn.datasets import load_files
print('加载数据')
t=time()
from new_file import get_json_list
from sklearn.feature_extraction.text import CountVectorizer

#语料
corpus = get_json_list()
#将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
#计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print (word)
print(len(word))
#查看词频结果
print (X.toarray())

# ----------------------------------------------------

from sklearn.feature_extraction.text import TfidfTransformer

#类调用
transformer = TfidfTransformer()
print (transformer)
#将词频矩阵X统计成TF-IDF值
tfidf_matrix = transformer.fit_transform(X)
#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
print (tfidf_matrix.toarray())

from sklearn.cluster import KMeans

num_clusters = 6

kmean=KMeans(n_clusters=num_clusters,
             max_iter=100,
             tol=0.1,
             verbose=1,
             n_init=3)

kmean.fit(X)
print("kmean: k={}, cost={}".format(num_clusters, int(kmean.inertia_)))
print("done in {0} seconds".format(time() - t))

#km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, init='k-means++',n_jobs=-1)
'''
n_clusters: 指定K的值
max_iter: 对于单次初始值计算的最大迭代次数
n_init: 重新选择初始值的次数
init: 制定初始值选择的算法
n_jobs: 进程个数，为-1的时候是指默认跑满CPU
注意，这个对于单个初始值的计算始终只会使用单进程计算，
并行计算只是针对与不同初始值的计算。比如n_init=10，n_jobs=40, 
服务器上面有20个CPU可以开40个进程，最终只会开10个进程
'''
#返回各自文本的所被分配到的类索引
#result = km_cluster.fit_predict(tfidf_matrix)
#print("Top terms per cluster:")

order_centroids = kmean.cluster_centers_.argsort()[:, ::-1]

#返回每个聚类最常出现的50个词

terms = vectorizer.get_feature_names()
result=[]
for i in range(num_clusters):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :50]:
        print(' %s' % terms[ind], end='')
    print()
