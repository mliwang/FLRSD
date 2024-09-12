# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:35:08 2021

@author: mliwang
"""

import json
import datetime
import os
import logging
import torch, random

from server import *
from client import *
from models import *
from datasets import *


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'	


from multiprocessing import  Process
import time

def multi_Process(candidates,model,A,Totaldatasize):
    print("start training based on boosted FL")
    threads = []           # 定义一个线程组
    for indexofclients,c in enumerate(candidates):
        threads.append(    # 线程组中加入赋值后的MyThread类
            MyProcess(indexofclients, c,model,A)  # 将每一个客户端的内容传到重写的MyThread类中
        )
    for thread in threads: # 每个线程组start
        thread.start()

    for thread in threads: # 每个线程组join
        thread.join()
        
    clients_num=len(candidates)

    aadiff = [0]*clients_num
    
    datasize=torch.zeros(clients_num)
    loss=torch.zeros(clients_num)
    for thread in threads:
        index,diff,lt =thread.get_result()
        aadiff[index]=diff
        datasize[index]=torch.tensor(float(len(thread.client.train_loader))/Totaldatasize,dtype=torch.float32)
        loss[index]=lt
        
    print("end")
    return aadiff, datasize, loss# 返回多线程返回的结果组成的列表
class MyProcess(Process):  # 重写Process类，加入获取返回值的函数

    def __init__(self, indexofclients, c,model,A):
        super(MyProcess,self).__init__()
        self.index = indexofclients                # 初始化传入的index
        self.client=c
        self.model=model
        self.A=A

    def run(self):                    # 新加入的函数，该函数目的：
        diff,lt =self.client.local_train(self.model,self.A)#第一个参数是梯度改变量，第二个是平均损失 # ①。调craw(arg)函数，并将初试化的url以参数传递——实现客户端训练
        self.diff=diff                             # ②。并获取craw(arg)函数的返回值存入本类的定义的值result中
        self.loss=lt 

    def get_result(self):  #新加入函数，该函数目的：返回run()函数得到的result
        return self.index,self.diff,self.loss
def main(conf):
	output_dir = "../%s/%s/source_%d" % (conf["dataset"], conf["undlying"], conf["num_source"])
	arg={}
	arg['indir']=  output_dir
	train_thread,test_thread,node_num=getdataForpro(conf["dataset"],conf["num_source"],conf["undlying"])
	train_datasets =G_TensorDataset(conf["dataset"],train_thread,conf["num_source"],arg=arg)
	eval_datasets =G_TensorDataset(conf["dataset"],test_thread,conf["num_source"],arg=arg)

	conf["node_num"]=node_num
	
	server = Server(conf, eval_datasets)
	clients = []
	
	for c in range(conf["no_models"]):#no_models  表明work的数量
		clients.append(Client(conf, server.global_model, train_datasets, c))
		
	print("\n\n",conf["dataset"]," num_source:",conf["num_source"]," underlying:",conf["undlying"])
	best_f1=0.0
	best_score={}

	s=open('num_source_%d'% conf["num_source"]+"underLying"+conf["undlying"]+"_"+conf["model_name"]+'_traing_log_%s.txt'% conf["dataset"],'w')
	mom_loss = torch.zeros(conf["global_epochs"],conf["no_models"])
	A=torch.tensor(getadjA__(arg),dtype=torch.float32)
	for e in range(conf["global_epochs"]):
		#这里要改，都参与了就把所有的都传上来
		candidates =random.sample(clients, conf["k"])#k表明每轮通信选几个机器的参数
		weight_accumulator = {}
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)

		#尝试把下面这段代码并行化
    #################顺序执行，用于调试##################
# 		datasize = torch.zeros(conf["no_models"])
# 		aadiff=[]
# 		for indexofclients,c in enumerate(candidates):
# 			diff,lt = c.local_train(server.global_model,A)#第一个参数是梯度改变量，第二个是平均损失

# 			aadiff.append(diff)
# 			datasize[indexofclients]=torch.tensor(float(len(c.train_loader))/len(train_datasets),dtype=torch.float32)
# 			mom_loss[e,indexofclients]=lt
        ############多线程并行化运行#########################
		totalsize=len(train_datasets)
		start = time.time()
		aadiff, datasize, loss_currentEpoch=multi_thread(candidates,server.global_model,A,totalsize)
		end = time.time()
		print("train time:",end-start)
		mom_loss[e]=loss_currentEpoch
		#############################################
		if e>1:
			#boosted FL
			mom_loss[e]=conf["beta"]*mom_loss[e-1]+(1-conf["beta"])*mom_loss[e]
		else:
			pass
		mom_loss[e]= datasize*mom_loss[e]
		#归一化权重
		bemean=mom_loss[e].mean()
		mom_loss[e]=mom_loss[e]-bemean
		maxml=mom_loss[e].max()
		mom_loss[e]=(torch.div(mom_loss[e],maxml)+1)*0.5

		summom=0
		for c in candidates:
			summom+=mom_loss[e][c.client_id]

		for candi in candidates:
			c=candi.client_id
			diff=aadiff[c]
			if conf["bfl"]=="bfl":
				w=mom_loss[e][c]/summom
			else:
				w=1/len(candidates)
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name].long()*w.long())
		server.model_aggregate(weight_accumulator)
		results = server.model_eval(A)
		if results['eval_f1']>best_f1:
			best_f1=results['eval_f1']
			best_score=results
		print("Epoch %d, acc: %f,precision: %f,recall:%f, f1: %f\n" % (e,results['accuracy'], results['precision'], results['recall'],results['eval_f1']))
		print("Epoch ",e,mom_loss[e]," loss:",loss_currentEpoch.mean(),"  eval_f1:",results['eval_f1'],"\n",file=s)
	s.close()
	print("best_f1",best_f1)
	f = open("BFLRSD_%s_%s_%d result.txt" %(conf["dataset"], conf["undlying"], conf["num_source"]), 'w')
	print("dataset:", conf["dataset"], 'popmodel:', conf["undlying"], "num_source:", conf["num_source"],"accuracy:", best_score['accuracy'],
		  'precision', best_score['precision'], 'Recall:', best_score['recall'],
					  'f1:', best_score['eval_f1'], 'error_distance:',
					  best_score['eval_error_distance'], "\n", file=f)
	f.close()
    


if __name__ == '__main__':

# 	parser = argparse.ArgumentParser(descion='Federated Learning')
# 	parser.add_argument('-c', '--conf', dest=ript'conf')
# 	args = parser.parse_args()
	confpath="conf.json"
	with open(confpath, 'r') as f:
		conf = json.load(f)	
	num_sources =[50, 100, 150, 200]#range(1,6)
	modelname=['GINConv']
	popmodel=["IC","SIR","SI"]

	for num_s in num_sources:
         for mod in popmodel:
             conf["num_source"]=num_s
             conf["undlying"]=mod
             main(conf)
	
