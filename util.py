# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:17:19 2021

@author: Administrator
"""
import datetime
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import coo_matrix 
#from kmeans_pytorch import kmeans
import networkx as nx
#from lign.utils.clustering import KMeans
# from cluster import KMeans
#from torch_geometric.utils.convert import to_networkx
#from torch_geometric.data import Data
import torch





def error_distance(A,pre,label):
    '''
    A 当前图的邻接矩阵  n n
    pre 预测的源头   []
    label 实际的源头 []
    算预测的源头和实际的源头之间的距离
    '''
#    coo_A = coo_matrix(A)
    G = nx.from_numpy_matrix(A)
    pn=np.where(pre==1)[0]
    lan=np.where(label==1)[0]
    # print("预测的源点：",pn)
    # print('实际的源点',lan)
    distance=0
    for p in pn:
        for i in lan:
            distance=distance+nx.shortest_path_length(G, source=int(p), target=int(i))
    d=0#abs(len(pn)-len(lan))
    return (distance+d)/(1+len(pn))
# def simplecluster(newgraph,num_clusters=5,device=torch.device("cpu")):
#     '''
#     一批图，利用其计算中心性
#     return 
#     C_s 各个图中各个聚类的介数中心值
#     T_mask 最可疑聚类
#     '''
#     C_s=[]#各个图中聚类中心性分数
#     T_mask=[]#各个图中最可疑的类的标记
#     c_id=[]#最可疑类的id
#     print('开始计算聚类中心性......')
#     for j in range(len(newgraph)):
#         cluster_ids_x,a1=newgraph[j]
# #        print(a1.size())
#         a1=a1.cpu().detach().numpy()
#         G = nx.Graph()
#         for s,t in zip(a1[0],a1[1]):
            
#             G.add_edge(s, t)
# #        G=to_networkx(g, to_undirected=True)
#         score = nx.betweenness_centrality(G)#  dict类型
#         print(score)
#         cluster_ids_x=cluster_ids_x.cpu()
#         ss=torch.FloatTensor([score[i] for i in range(len(cluster_ids_x))]).unsqueeze(1)#中心性分数向量
#         cluster_score=[]#聚类分数
#         for i in range(num_clusters):
#             seletx_i=torch.eq(cluster_ids_x,i).unsqueeze(1).float()#n,1
#             s_i=torch.mm(ss.t(),seletx_i).squeeze(1)#拿到当前类的中心性分数，实际是当前类中所有点中心性的总和
#             s_i=torch.div(s_i,torch.sum(seletx_i,dim=0))
#             cluster_score.append(s_i)
#         cluster_score=torch.stack(cluster_score)#num_clusters,1
#         C_s.append(cluster_score)
#         #得到中心值最大的类的所有点位置的标记，属于可疑类的位置标记为1，max_cluster_mask->n*1
#         max_cluster_id=torch.argmax(cluster_score,0)#目标类的id (1,)
#         c_id.append(max_cluster_id)
#         max_cluster_mask=torch.eq(cluster_ids_x,max_cluster_id).unsqueeze(1).float()
#         T_mask.append(max_cluster_mask)
#     C_s=torch.stack(C_s)# batch-size,num_clusters,1
#     T_mask=torch.stack(T_mask)# batch-size,n,1
#     c_id=torch.stack(c_id)# batch-size,1
#     return C_s.to(device),T_mask.to(device),c_id.to(device)

# def cluster(b,A,num_clusters=5,device=torch.device("cpu")):
#     '''
#     聚类并拿到各个类的中心性
#     b为一批节点特征
#     A为一批邻接矩阵
#     num_clusters 为聚类的数量
    
#     return 
#     C_s 各个图中各个聚类的介数中心值
#     T_mask 最可疑聚类
#     '''
#     C_s=[]#各个图中聚类中心性分数
#     T_mask=[]#各个图中最可疑的类的标记
#     c_id=[]#最可疑类的id
#     A=A.cpu().detach()
#     for i,x in enumerate(b):
#         x=x.cpu()
#         # kmeans
#         print('开始聚类......')
#         kmeans = KMeans(x, k=num_clusters, n_iters =100)
# #        kmeans.to(device)
#         cluster_ids_x =kmeans(x) # 预测分类
#         print('聚类完成')
# #        cluster_ids_x=cluster_ids_x.cpu()
#         #拿到中心节点的聚类中心性来代表当前聚类
#         coo_A = coo_matrix(A[i].numpy())
#         G = nx.Graph()
#         for s,t in zip(coo_A.row, coo_A.col):
#             G.add_edge(s, t, weight=A[i][s][t].numpy())
        
#         score = nx.betweenness_centrality(G)#  dict类型
# #        print(score)
        
#         ss=torch.FloatTensor([score[i] for i in range(len(cluster_ids_x))]).unsqueeze(1)#中心性分数向量
#         cluster_score=[]#聚类分数
#         for i in range(num_clusters):
#             seletx_i=torch.eq(cluster_ids_x,i).unsqueeze(1).float()#n,1
#             s_i=torch.mm(ss.t(),seletx_i).squeeze(1)#拿到当前类的中心性分数，实际是当前类中所有点中心性的总和
#             s_i=torch.div(s_i,torch.sum(seletx_i,dim=0))
#             cluster_score.append(s_i)
#         cluster_score=torch.stack(cluster_score)#num_clusters,1
#         C_s.append(cluster_score)
#         #得到中心值最大的类的所有点位置的标记，属于可疑类的位置标记为1，max_cluster_mask->n*1
#         max_cluster_id=torch.argmax(cluster_score,0)#目标类的id (1,)
#         c_id.append(max_cluster_id)
#         max_cluster_mask=torch.eq(cluster_ids_x,max_cluster_id).unsqueeze(1).float()
#         T_mask.append(max_cluster_mask)
#     C_s=torch.stack(C_s)# batch-size,num_clusters,1
#     T_mask=torch.stack(T_mask)# batch-size,n,1
#     c_id=torch.stack(c_id)# batch-size,1
#     return C_s.to(device),T_mask.to(device),c_id.to(device)

class MyGraph(object):
    '''
    适用于边异质的图
    '''
    def __init__(self,n_g,g_n):
        '''
        n_g 原始id到图中id的映射字典
        g_n 图中id到原始id的映射字典
        '''
#        self.edgedict=edgedict#输入边邻接矩阵列表
        self.n_g=n_g#原始id到图中id的映射
        self.g_n=g_n#图中id到原始id的映射
        self.node_num=len(n_g)#涉及到的节点数
        
    def getadju(self,di):
        '''
        由链表得到邻接矩阵
        di 邻接链表
        
        '''
#        print(self.node_num)
        Adju=np.zeros((self.node_num,self.node_num))
        if di==None:
            return np.random.randint(0,5,(self.node_num,self.node_num))
#            return Adju
        
        for key, item in di.items():
            if item is None or len(item)<1:
                pass
            s=self.n_g[key]#源节点
            for i in item:
                t=self.n_g[i]#目标节点
                Adju[s][t]=Adju[s][t]+1
        return Adju
#    def getAdjNormornize(self,A):
#        '''
#        矩阵归一化
#        '''
#        R = np.sum(A, axis=1)
#        R_sqrt = 1/np.sqrt(R) 
#        D_sqrt = np.diag(R_sqrt)
#        I = np.eye(A.shape[0])
#        return I - D_sqrt * A * D_sqrt
    
    
    def csv_dict(self,data,source_c,target_c):
        '''
        把关系表转换成dict
        '''
        return data.groupby(source_c)[target_c].apply(list).to_dict()
    
    def process_Adjacent(self,event):
        '''
        加载数据（用户转发矩阵、朋友关系矩阵），并处理成numpy.array,归一化
        由于用户评论关系各个thread情况不同，所以作为动态部分，在训练前处理
        '''
        with open("middle/tweets/user_friendship/%s.pkl"% event, 'rb') as f:
            friendships=pickle.load(f)
#        A_f=self.getAdjNormornize(self.getadju(friendships))
        A_f=self.getadju(friendships)
#        A_r=None
#        with open("middle/tweets/retweet_friendship_%s.pkl"% event, 'rb') as f:
#            retweets=pickle.load(f)
        retweets=None
#        A_r=self.getAdjNormornize(self.getadju(retweets))
        A_r=self.getadju(retweets)
        return A_f,A_r
    



def fetch_tweets(event):
    """ Read a CSV file with cleaned PHEME event tweets
    
    Note: 
        - Setting engine to "python" helps with large datasets
    
    Params:
        - event {str} the name of the event
    
    Return: a Pandas dataframe
    
    """
#    strc=['tweet_id','in_reply_tweet','thread','user_id','in_reply_user','created']
#    pd.set_option('display.float_format',lambda x: '%.0f' % x)
    df=pd.read_csv("middle/tweets/%s.csv" % event, 
#                 dtype={
#                    'tweet_id': str,
#                    'in_reply_tweet': str,
#                    'thread': str,
#                    'user_id': str,
#                    'in_reply_user': str
#                 },
                 engine="python")
    strc=['tweet_id','in_reply_tweet','thread','user_id','in_reply_user']
    for c in strc:
        df[c]=df[c].apply('{:1f}'.format).astype(str).apply(lambda x: x.split('.')[0])
    return df

def to_unix_tmsp(col):
    """ Convert Datetime instance to Unix timestamp """
    return pd.DatetimeIndex(col).astype(np.int64) / 1e6

def parse_twitter_datetime(timestr):
    """ Convert Twitter's datetime format into a Datetime instance """
    return pd.datetime.strptime(timestr, "%a %b %d %H:%M:%S %z %Y")

def fetch_X(thread_level_csv_file_address):
    """ Read a CSV file with thread-level features and drop all column that are not used in prediction.
    
    Note: 
        - Setting engine to "python" helps with large datasets
    
    Params:
        - event {str} the name of the event
    
    Return: a Pandas dataframe
    
    """
    X= pd.read_csv(thread_level_csv_file_address,engine="python")
    if 'event' in X.columns:
        X=gw_thrds_without_rumor_tag=X.drop(['event'],axis=1)
    return X

def fetch_thread(event, is_normalized=True):
    """ Return dataset X and results vector y 
    
    Params:
        - event {str} the name of the event in the PHEME dataset
        - is_normalized {bool} returned X matrix as normalized. Deafult is True
    """ 
    X = pd.read_csv("data/threads/%s.csv" % event, engine="python")
    y = X.is_rumor
    X = X.drop(["is_rumor", "thread"], axis=1)
    if is_normalized:
        X = (X - X.mean()) / X.std()
    return X, y