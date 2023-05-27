# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 20:00:54 2021

@author: mliwang
"""
import torch
import numpy as np
import pickle
import random
from torch.utils.data import Dataset
def getAdjNormornize(A):
    '''
    矩阵归一化,Laplace
    '''
    R = np.sum(A, axis=1)
    R_sqrt = 1/np.sqrt(R) 
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(A.shape[0])
    return I - D_sqrt * A * D_sqrt


def getdataForpro(dataset, num_source, popmodel, train_pro=0.9):
    output_dir = "../%s/%s/source_%d" % (dataset, popmodel, num_source)
    with open(output_dir + "/adj.pkl", 'rb') as f:
        A = pickle.load(f)
    Threadlists = [i for i in range(1000)]
    random.shuffle(Threadlists)
    train_thread = Threadlists[:int(len(Threadlists) * train_pro)]
    test_thread = list(filter(lambda x: x not in train_thread, Threadlists))
    print('训练样本', len(train_thread), '测试样本:', len(test_thread))

    return train_thread, test_thread, A


class G_TensorDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, datasetname, thread, global_graph, num_source, inputdir, istrain=True, alpha=0.4):
        self.istrain = istrain

        self.datasetname = datasetname
        self.thread = thread
        self.alpha = alpha
        self.num_source = num_source
        self.datadir = inputdir

        self.A = self.graph = np.array(global_graph)  # 邻接矩阵

        self.S = getAdjNormornize(self.A)
        # 填空值
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.S = imp.fit_transform(self.S)

    def __getitem__(self, index):
        '''
        把数据处理成样本 sample:(S,X,Y)
        S 邻接矩阵
        X 用户节点特征矩阵
        state 用户传播谣言状态，（即用户是否发推,用户传播了谣言为1，没有传播为-1）
        spread_time  归一化的谣言散布时间（虽然经过处理但是时间的先后顺序没变）
        Y 当前用户是否是源头

        '''
        #        print(index)
        batch = self.thread[index]  # 一个thread 列表
        with open(self.datadir + "/Thread_%d.pkl" % batch, 'rb') as f:
            data = pickle.load(f)
        '''
        {'source_id':list_s,
                         'net_state':net_state,
                         }, 
        '''

        # 构造state
        state = -np.ones((len(self.graph), 1))

        v34 = np.zeros((len(self.graph), 2))
        # u_sum_in[np.where(u_sum_in == 0)] = 1
        for u, s in enumerate(data['net_state']):
            if s == 1:
                state[u][0] = 1
                v34[u][1] = 0
                v34[u][0] = 1
            else:
                state[u][0] = -1
                v34[u][0] = 0
                v34[u][1] = -1
        d2 = np.dot((1 - self.alpha) * np.linalg.inv(np.eye(state.shape[0]) - self.alpha * self.S), state)
        X = np.hstack((state, d2))
        d3 = np.dot((1 - self.alpha) * np.linalg.inv(np.eye(state.shape[0]) - self.alpha * self.S), v34[:, 0])
        X = np.hstack((X, np.expand_dims(d3, axis=1)))
        d4 = np.dot((1 - self.alpha) * np.linalg.inv(np.eye(state.shape[0]) - self.alpha * self.S), v34[:, 1])
        X = np.hstack((X, np.expand_dims(d4, axis=1))).astype(float)
        #        print('X:*******',X)
        #            print(static_f.shape,'state',state.shape,'   X.shape',X.shape)

        if self.istrain:
            # 构造Y
            Y = np.zeros((len(self.graph), 1))
            # 找到源头用户
            for u in data['source_id']:
                Y[u][0] = 1

            #            Y=self.graph.n_g[temp[temp['is_source_tweet']==1]['user_id'].unique()[0]]
            return (
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(state, dtype=torch.float32),
                torch.tensor(Y))
        else:
            return (
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(state, dtype=torch.float32)
            )

    def __len__(self):
        return len(self.thread)
