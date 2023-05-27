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

if __name__ == '__main__':

# 	parser = argparse.ArgumentParser(description='Federated Learning')
# 	parser.add_argument('-c', '--conf', dest='conf')
# 	args = parser.parse_args()
	confpath="conf.json"
	with open(confpath, 'r') as f:
		conf = json.load(f)	
	output_dir = "../%s/%s/source_%d" % (conf["dataset"], conf["undlying"], conf["num_source"])
	train_thread,test_thread,A=getdataForpro(conf["dataset"],conf["num_source"],conf["undlying"])
	train_datasets =G_TensorDataset(conf["dataset"],train_thread,A,conf["num_source"],output_dir)
	eval_datasets =G_TensorDataset(conf["dataset"],test_thread,A,conf["num_source"],output_dir)

	conf["node_num"]=len(A)
	
	server = Server(conf, eval_datasets)
	clients = []
	
	for c in range(conf["no_models"]):#no_models  表明work的数量
		clients.append(Client(conf, server.global_model, train_datasets, c))
		
	print("\n\n")
	best_f1=0.0
	best_score={}

	s=open('num_source_%d'% conf["num_source"]+"_"+conf["model_name"]+'_traing_log_%s.txt'% conf["dataset"],'w')
	mom_loss = torch.zeros(conf["global_epochs"],conf["no_models"])    
	for e in range(conf["global_epochs"]):
		#这里要改，都参与了就把所有的都传上来
		candidates =clients #random.sample(clients, conf["k"])#k表明每轮通信选几个机器的参数
		weight_accumulator = {}
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		datasize=torch.zeros(conf["no_models"])
		#尝试把下面这段代码并行化
		aadiff=[]
		for indexofclients,c in enumerate(candidates):
			diff,lt = c.local_train(server.global_model,A)#第一个参数是梯度改变量，第二个是平均损失
			aadiff.append(diff)
			datasize[indexofclients]=torch.tensor(float(len(c.train_loader))/len(train_datasets),dtype=torch.float32)
			mom_loss[e,indexofclients]=lt
		if e>0:
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
		for c in range(len(candidates)):
			diff=aadiff[c]
			w=mom_loss[e][c]
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name].long()*w.long())
		server.model_aggregate(weight_accumulator)
		results = server.model_eval(A)
		if results['eval_f1']>best_f1:
			best_f1=results['eval_f1']
			best_score=results
		print("Epoch %d, acc: %f,precision: %f,recall:%f, f1: %f\n" % (e,results['accuracy'], results['precision'], results['recall'],results['eval_f1']))
		print(results['eval_f1'],"\n",file=s)	
	s.close()
	print("best_f1",best_f1)
	f = open("BFLRSD_%s_%s_%d result.txt" %(conf["dataset"], conf["undlying"], conf["num_source"]), 'w')
	print("dataset:", conf["dataset"], 'popmodel:', conf["undlying"], "num_source:", conf["num_source"],"accuracy:", best_score['accuracy'],
		  'precision', best_score['precision'], 'Recall:', best_score['recall'],
					  'f1:', best_score['eval_f1'], 'error_distance:',
					  best_score['eval_error_distance'], "\n", file=f)
	f.close()