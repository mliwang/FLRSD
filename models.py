# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:44:46 2021

@author: mliwang
"""

import math
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from collections import OrderedDict
from torch.nn.modules.module import Module
from torch_geometric.nn import GraphConv,SAGEConv,GCNConv,GATConv,TAGConv,GINConv

from kmeans import lloyd
class GCN(nn.Module):
    def __init__(self,modelname, nfeat, nhid, nclass, dropout):
        '''
        modelname:使用的模型名称
        nfeat:输入x特征矩阵维度
        nhid：中间层维度
        nclass：输出特征维度
        dropout：dropout的比例
        ['GNN','GCN','GAT','GraphSAGE','TAGCN']
        '''
        super(GCN, self).__init__()
        if modelname=='GNN':
            self.gc1 = GraphConv(nfeat, nhid)
            self.gc2 = GraphConv(nhid, nclass)
        elif modelname=='GCN':
            self.gc1 = GCNConv(nfeat, nhid)
            self.gc2 = GCNConv(nhid, nclass)
        elif modelname=='GAT':
            self.gc1 = GATConv(nfeat, nhid)
            self.gc2 = GATConv(nhid, nclass)
        elif modelname=='GraphSAGE':
            self.gc1 = SAGEConv(nfeat, nhid)
            self.gc2 = SAGEConv(nhid, nclass)
        elif modelname=='TAGCN':
            self.gc1 = TAGConv(nfeat, nhid)
            self.gc2 = TAGConv(nhid, nclass)
        elif modelname=='GINConv':
            self.gc1 = GINConv(self.MLP(nfeat, nhid))
            self.gc2 = GINConv(self.MLP(nhid, nclass))
        self.dropout = dropout
    @staticmethod
    def MLP(in_channels: int, out_channels: int) -> torch.nn.Module:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)
            


class Mycluster(nn.Module):
    def __init__(self, modelname,nfeat, hidden, nclass):
        '''
        nfeat:输入x特征矩阵维度
        hidden：中间层维度
        nclass：分类数
        
        '''
        super(Mycluster, self).__init__()
        if modelname=='GNN':
            self.gc1 = GraphConv(nfeat, hidden)
        elif modelname=='GCN' or modelname=='GINConv':
            self.gc1 = GCNConv(nfeat, hidden)
        elif modelname=='GAT':
            self.gc1 = GATConv(nfeat, hidden)
        elif modelname=='GraphSAGE':
            self.gc1 = SAGEConv(nfeat, hidden)
        elif modelname=='TAGCN':
            self.gc1 = TAGConv(nfeat, hidden)

        self.mlp2=nn.Linear(hidden, nclass)
        

    def forward(self, x,adj):
        x = F.relu(self.gc1(x, adj))
        x=self.mlp2(x)
        return F.relu(x)    #n,nclass
# class Centrility(nn.Module):
#     def __init__(self, nfeat, hidden,node_num):
#         '''
#         nfeat:输入x特征矩阵维度
#         hidden：中间层维度
#         node_num：节点数
        
#         '''
#         super(Centrility, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, hidden)
#         self.mlp1=nn.Linear(node_num*hidden, node_num)
#         self.mlp1.apply(self._init_weights)
        
#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=0.002)


#     def forward(self, x,adj):
#         x = self.gc1(x, adj)
#         x=torch.flatten(x,0)
#         x=self.mlp1(x)
#         return F.log_softmax(x,dim=-1)    #n,nclass


class RumorDetect(nn.Module):
    def __init__(self, modelname, node_num, nfeat, nhid, nclass, dropout, device, num_clusters=5, hidden=256):
        '''
        node_num图中节点总数
        nfeat:输入x特征矩阵维度
        nhid：中间层维度
        nclass：输出特征维度
        dropout：dropout的比例

        '''
        super(RumorDetect, self).__init__()

        # self.w1=nn.Linear(node_num, node_num)
        # self.w2=nn.Linear(node_num, node_num)
        # self.w3=nn.Linear(node_num, node_num)

        self.gcn = GCN(modelname, nfeat, nhid, nclass, dropout)
        self.norm = nn.BatchNorm1d(nclass + nfeat)

        self.norm1 = nn.BatchNorm1d(node_num)
        self.mlp1 = nn.Linear(node_num, 32 * 4)  # 用于最后输出

        self.norm2 = nn.BatchNorm1d(32 * 4)
        self.mlp2 = nn.Linear(32 * 4, node_num * 2)

        self.mycluster = Mycluster(modelname, nclass + nfeat, hidden, num_clusters)
        self.clusternorm1 = nn.BatchNorm1d(num_clusters)

        self.Centrility = nn.Linear(nclass + nfeat, 1)
        self.num_clusters = num_clusters
        self.dropout = dropout
        self.device = device
        self.node_num = node_num
        self.k = 2

    def forward(self, x, state, a1, a2=None, a3=None, kmeans=False, Centrilityname="us"):
        '''
       x  特征矩阵  b,n,nfeat
       a1, 朋友关系矩阵 b,n,n
       a2,转推关系矩 b,n,n
       a3 评论关系矩 b,n,n
       A=a1W1+a2W2+a3W3

       a 超图邻接矩阵
       x, 经过学习的节点特征矩阵
       state_orginal  节点原始状态 b,n,1
       '''
        #        print('a1:',a1)
        #        print('a2:',a2)
        #        print('a3:',a3)
        if a2 == None:
            adj = a1  # F.relu(a1)
        else:
            adj = F.relu(a1 + a2 + a3)
        a1 = (adj > 0).nonzero().t().to(self.device)

        # adj=F.relu(self.w1(a1)+self.w2(a2)+self.w3(a3))
        #        adj=self.normalize_adj_torch(adj)
        out_seq = []
        #        newgraph=[]
        #        cluster_ids=[]
        C_s = []  # 各个图中聚类中心性分数
        T_mask = []  # 各个图中最可疑的类的标记
        c_id = []  # 最可疑类的id
        # A_new=[]
        for t, embs in enumerate(x):
            # print('Ahat',node_embs)

            # A_new.append(a)
            #            print('model_size',a1.size())
            node_embs = self.gcn(embs, a1)  # n,class
            node_embs = torch.cat((embs, node_embs), 1)  # n,(class+nfeat)
            node_embs = self.norm(node_embs)
            # 聚类
            if kmeans:
                cluster_ids_x, _ = lloyd(node_embs, self.num_clusters, self.device)

            else:

                myclusters_x = self.mycluster(node_embs, a1)  # 预测分类n,cluster
                myclusters_x = self.clusternorm1(myclusters_x)

                cluster_ids_x = torch.argmax(F.softmax(myclusters_x, dim=1), dim=1)  # n
            #            print('myclusters_x.size:',myclusters_x)
            # 计算图的中心性
            #            ss=torch.div(Ahat.sum(dim=0),Ahat.size()[1]).unsqueeze(1)#度中心性torch.div(a.sum(dim=0),a.size()[1])

            # 得到各个节点的中心性分数
            if Centrilityname == "degree":
                ss = torch.div(adj.sum(dim=0), adj.size()[1]).unsqueeze(1)  # adj
            elif Centrilityname == "eig":  # 特征向量中心性
                evals, _ = torch.eig(adj, eigenvectors=False)
                ss = evals[:, 0].unsqueeze(1)

            else:
                # 本文设计的中心性计算
                ss = F.sigmoid(self.Centrility(node_embs))  # [34, 1]
            # print("ss",ss.size())

            cluster_score = []  # 聚类分数
            for i in range(self.num_clusters):
                seletx_i = torch.eq(cluster_ids_x, i).unsqueeze(1).float()  # n,1
                s_i = torch.mm(ss.t(), seletx_i).squeeze(1)  # 拿到当前类的中心性分数，实际是当前类中所有点中心性的总和
                #                print('中间变量',s_i)
                s_i = torch.div(s_i, torch.sum(seletx_i, dim=0).clamp_(1))
                # print('单个聚类的中心性s_i：',s_i)
                cluster_score.append(s_i)
            cluster_score = torch.stack(cluster_score)  # num_clusters,1
            C_s.append(cluster_score)

            # 得到中心值最大的类的所有点位置的标记，属于可疑类的位置标记为1，max_cluster_mask->n*1
            #            max_cluster_id=torch.argmax(cluster_score,0)#目标类的id (1,)
            max_cluster_id = torch.topk(cluster_score, self.k, 0)[1].squeeze(1)
            c_id.append(max_cluster_id)
            max_cluster_mask = torch.zeros_like(cluster_ids_x)  # torch.zeros(cluster_ids_x.size()[0],1).to(self.device)
            for c_i in max_cluster_id:
                max_cluster_mask = max_cluster_mask + torch.eq(cluster_ids_x, c_i).float()
            max_cluster_mask = max_cluster_mask.unsqueeze(1)
            # print('max_cluster_mask',max_cluster_mask)
            T_mask.append(max_cluster_mask)
            out_seq.append(node_embs)
            del node_embs, cluster_ids_x, ss, cluster_score, seletx_i, s_i, max_cluster_id, max_cluster_mask
            gc.collect()
        x = torch.stack(out_seq)
        del out_seq
        gc.collect()
        C_s = torch.stack(C_s)  # batch-size,num_clusters,1
        T_mask = torch.stack(T_mask)  # batch-size,n,1
        c_id = torch.stack(c_id)  # batch-size,1

        s = F.relu(torch.matmul(x, x.transpose(1, 2)))  # 节点间传播概率 b,n,n
        output = torch.matmul(s, state).squeeze(2)  # 原始节点感染状态 output b,n,1
        output = self.norm1(output)
        output = F.relu(self.mlp1(output))
        output = self.norm2(output)

        output = self.mlp2(output)
        output = F.sigmoid(output.view(-1, self.node_num, 2))
        return C_s, T_mask, c_id, output  # ,A_new


def get_model(conf,device):
    num_cluster=5
    if conf['dataset']=='public_karate':
        num_cluster=5
    elif conf['dataset']=='facebook':
        num_cluster=30
    elif conf['dataset']=='weibo':
        num_cluster = 50

    return RumorDetect(conf['model_name'],conf['node_num'],conf['nfeat'], conf['nhid'], conf['nclass'], conf['dropout'],device,num_cluster)
