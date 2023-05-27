# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:43:31 2021

@author: mliwang
"""

import models, torch
import numpy as np
import gc
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class Client(object):

	def __init__(self, conf, model, train_dataset, id = -1):
		
		self.conf = conf
		
		self.local_model =models.get_model(self.conf,device) 
		
		self.client_id = id
		
		self.train_dataset = train_dataset
		
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset) / self.conf['no_models'])
		if id<self.conf['no_models']-1:
		    train_indices = all_range[id * data_len: (id + 1) * data_len]
		else:
		    train_indices = all_range[id * data_len:]

		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
									sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
									
		
	def local_train(self, model,A):

		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	
		#print(id(model))
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		ww=torch.from_numpy(np.array([self.local_model.node_num,2])).float()
		loss_fn=torch.nn.CrossEntropyLoss()#torch.nn.CrossEntropyLoss(weight=ww,size_average=True)
		#loss_fn1=torch.nn.CrossEntropyLoss()
		loss_fn.to(device)
		#print(id(self.local_model))

		self.local_model.train()
		A=torch.tensor(A,dtype=torch.float32) 
		localloss=[]
		for e in range(self.conf["local_epochs"]):
			optimizer.zero_grad()
			total_loss=0.0
			for batch_id, batch in enumerate(self.train_loader):
				X,state,Y =(x.to(device) for x in batch)
				C_s,T_mask,c_id,state_orginal=self.local_model(X,state,A)#state_orginal ,b,n,2
				del batch,state
				gc.collect()
				pred=torch.mul(T_mask.repeat(1,1,2),state_orginal)#这样就把其他聚类给屏蔽了

# 				loss=[]
				#loss1=[]
# 				for i in range(len(X)):
# # 					print("pre  ",pred[i])
# # 					print("groundtruth  ",Y[i])
# 					tem_loss=loss_fn(pred[i] ,Y[i].squeeze(1).long())
# 					#t_l=loss_fn1(pred[i] ,Y[i].squeeze(1).long())
# 					#loss1.append(t_l)
# 					loss.append(tem_loss)
				#loss = torch.nn.functional.cross_entropy(pred.view(pred.size()[0]*X.size()[1],-1) ,Y.view(Y.size()[0]*X.size()[1]).long())
				onebatchloss=[]
# 				print("pred.size():",pred.size())
# 				print("Y.size():",Y.size())
				for b,y in zip(pred,Y): 
				    onebatchloss.append(loss_fn(b,y.squeeze(1).long()))
                    
				loss=torch.stack(onebatchloss)
				#loss1=torch.stack(loss1)
				loss=torch.mean(loss)
				#loss1=torch.mean(loss1)
				# print("loss:",loss)
				loss.backward()
				total_loss = loss.item() if total_loss is None else total_loss + loss.item()
			
				optimizer.step()
			# print("Epoch: ",e,"  done, loss:" ,total_loss)
			localloss.append(total_loss)
		print("localloss:",localloss)
		localloss=np.array(localloss)
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			#print(diff[name])
			
		return diff,torch.tensor(localloss.mean(),dtype=torch.float32)