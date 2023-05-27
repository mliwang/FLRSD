# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:36:20 2021

@author: mliwang
"""

import models, torch
from util import *
import torch.nn.functional as F
import gc
import numpy as np
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class Server(object):
	
	def __init__(self, conf, eval_dataset):
	
		self.conf = conf
		
		
		self.global_model = models.get_model(self.conf,device) 
		
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)
				
	def model_eval(self,A1):
		self.global_model.eval()
		
		dataset_size = 0
		pre_user=[]#
		Groud_user=[]#
		# A=torch.tensor(A1,dtype=torch.float32)
		for batch_id, batch in enumerate(self.eval_loader):
			X,state,Y =(x.to(device) for x in batch)
			Groud_user.append(Y.squeeze(2).cpu().numpy()) #b,n
# 			data, target = batch 
			dataset_size += X.size()[0]
			del batch
			with torch.no_grad():
				_,T_mask,_,state_orginal= self.global_model(X,state,A1)
				state_orginal=F.softmax(state_orginal,dim=2)
				pred=torch.mul(T_mask.repeat(1,1,2),state_orginal)
				source_user=torch.argmax(pred,dim=2)#b,n
				pre_user.append(source_user.cpu().numpy())# batch_size,topk
			del X
			gc.collect()
		pre_user=np.concatenate(pre_user,0)
		Groud_user=np.concatenate(Groud_user,0)

		# print("shape of A:",A.shape)
		result=eval(pre_user,Groud_user,A1)

		return result
def calulateF(pre,label):
	pn=np.where(pre==1)[0]
	lan=np.where(label==1)[0]
	count=0
	for p in pn:
		if p in lan:
			count += 1
	precision = count / len(lan)
	recall = count /(len(pn)+1)
	f_score = (2 * precision * recall) / (precision + recall+ 0.001)
	return precision,recall,f_score


def eval(pre_user,Groud_user,A_graph):
	'''
	Groud_user  len ,n
	pre_user len n
	A_graph n,n
	'''
	#æ±åºacc
	from sklearn.metrics import accuracy_score
	results={}
	accu=0
	precision=0
	recall=0
	f1_micro=0
	dis=0
	for i,t in enumerate(Groud_user):
		accu=accu+accuracy_score(t,pre_user[i])
		pt,rt,ft=calulateF(pre_user[i],t)
		precision=precision+pt#precision_score(t,pre_user[i])
		recall=recall+rt#recall_score(t,pre_user[i])
		f1_micro=f1_micro+ft#f1_score(t,pre_user[i])
		
		#dis=dis+error_distance(A_graph,pre_user[i],t)
	results['accuracy']=accu/(1.0 * len(Groud_user))
	results['precision']=precision/(1.0 * len(Groud_user))
	results['recall']=recall/(1.0 * len(Groud_user))
	results['eval_f1']=f1_micro/(1.0 * len(Groud_user))
	results['eval_error_distance']=dis/(1.0 * len(Groud_user))
	return results