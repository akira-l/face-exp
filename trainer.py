#!/usr/bin/env python3
# coding: utf-8

from __future__ import print_function
from __future__ import division

import os
import datetime
import math
import argparse
import numpy as np
import os.path as osp
import json
import pdb

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms


from retrieval_test_data import face_a_dis_valid

#from utils.logger import Logger


class Trainer():
	def __init__(self, net, face_loss, train_loader, test_loader, optimizer, start_epoch=0,
				 best_accuracy=0, best_epoch=0, base_lr=0.0001, criterion=nn.CrossEntropyLoss(),
				 lr_decay_interval=50, use_cuda=True, save_dir='checkpoint/resnet18/',
				 totalEpoch=7, historyDataFile="checkpoint/resnet18/historyData.t7", outputInterval=20):
		self.totalEpoch = totalEpoch
		self.outputInterval=outputInterval
		self.historyDataFile=historyDataFile
# length = total epoches
		self.historyDict = {
			"meanTrainLoss": np.zeros((totalEpoch),dtype='float32'),
			"meanTrainAcc": np.zeros((totalEpoch),dtype='float32'),
			"meanTestLoss": np.zeros((totalEpoch),dtype='float32'),
			"meanTestAcc": np.zeros((totalEpoch),dtype='float32'),
		}
		self.net = net
		self.train_loader = train_loader
		self.test_loader = test_loader
		transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
		])
		trainset_valid = face_a_dis_valid(root="./../face_a/train",
                      list_file = "./../face_a/train.csv",
                      train=False, 
                      transform=transform, 
                      input_size=224)
		self.trainloader_valid = torch.utils.data.DataLoader(trainset_valid, 
                                          batch_size=100, 
                                          shuffle=False, 
                                          num_workers=0, 
                                          collate_fn=trainset_valid.collate_fn)

		self.optimizer = optimizer

		self.base_lr = base_lr
		self.criterion = criterion
		self.lr_decay_interval = lr_decay_interval
		self.use_cuda = use_cuda

		self.best_accuracy = best_accuracy
		self.best_epoch = best_epoch
		self.start_epoch = start_epoch
		self.save_dir = save_dir
		self.MCP = face_loss
		self.cosine_dis = torch.nn.CosineSimilarity()

		'''
		try:
			from tools.logger import Logger
		except ImportError as e:
			print("fail to import tensorboard: {} ".format(e))
		else:
			self.tflog_writer = Logger(self.save_dir, restart=True)
		'''
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

		self.jsonlog_writer_train = open(osp.join(self.save_dir, "train.log"), 'w+') # for what?
		self.jsonlog_writer_test = open(osp.join(self.save_dir, "test.log"), 'w+')

	def __del__(self):
		self.jsonlog_writer_train.close()
		self.jsonlog_writer_test.close()

	def train(self, epoch):
		""" Traning epoch """
		print('==> Training Epoch: %d' % epoch)
		self.net.train()
		total_train_loss = 0
		total_correct = 0
		total_size = 0

		n_train = len(self.train_loader.dataset)
		for batch_idx, (inputs, targets, imgFileName) in enumerate(self.train_loader):
			if self.use_cuda:
				inputs, targets = inputs.cuda(), targets.cuda()
			inputs, targets = Variable(inputs), Variable(targets)
			self.optimizer.zero_grad()
			outputs = self.net(inputs)
			
			if epoch < 15:
    				id_feature = outputs
			else:
    				id_feature = self.MCP(outputs, targets)
			
			#pdb.set_trace()
			
			loss = self.criterion(id_feature, targets)
			loss.backward()
			self.optimizer.step()

			total_train_loss += loss.item() # scalar
			_, predicted = torch.max(outputs.data, dim=1)
			batch_correct = predicted.eq(targets).sum()
			total_correct += batch_correct
			error_rate = 1 - float(batch_correct) / len(inputs)
			total_size += targets.size(0)

			if (batch_idx % self.outputInterval == 0):
				print('Epoch: [{}]\tTrain:[{}/{} ({:.2f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
					epoch, total_size, n_train,   1e2*total_size/n_train,
					loss.item(), error_rate ))
# 			break

		meanTrainLoss = total_train_loss/len(self.train_loader) # scalar
		meanTrainAcc = 1e2* float(total_correct) /n_train # scalar
		info = {
				'epoch': epoch, # scalar
				'train-loss': meanTrainLoss, # scalar
				'train-top1-acc': meanTrainAcc # scalar
		}
		self.jsonlog_writer_train.write(json.dumps(info) + "\n")
		print("Epoch:%d\t Mean train loss:%.6f,\tMean train acc:%.6f%%"%( epoch, 
					      meanTrainLoss,         meanTrainAcc) ) # n_train= number of total images in train set
		self.historyDict["meanTrainLoss"][epoch] = meanTrainLoss
		self.historyDict["meanTrainAcc"][epoch] = meanTrainAcc

	def test(self, epoch):
		""" Testing epoch """
		print('==> Testing Epoch: %d' % epoch)
		self.net.eval()
		total_test_loss = 0
		total_correct = 0
		total_size = 0
		n_test = len(self.test_loader.dataset) # ->n_test
		for batch_idx, (inputs, targets, imgFileName) in enumerate(self.test_loader):
			if self.use_cuda:
				inputs, targets = inputs.cuda(), targets.cuda()
			inputs, targets = Variable(inputs, requires_grad=False), Variable(targets, requires_grad=False)
			with torch.no_grad():
				outputs = self.net(inputs)
			loss = self.criterion(outputs, targets)

			total_test_loss += loss
			_, predicted = torch.max(outputs, dim=1)
			batch_correct = predicted.eq(targets).sum()
			total_correct += batch_correct
			total_size += targets.size(0)
			error_rate = 1 - float(batch_correct) / len(inputs)

#			partial_epoch = epoch + batch_idx / len(self.train_loader) # what's partial_epoch used for?
			if (batch_idx % self.outputInterval == 0):
				print('Epoch: [{}]\tTest: [{}/{} ({:.2f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
					epoch,                total_size,n_test, 1e2*total_size/n_test,
					loss.item(), error_rate ))

#		print('Epoch: [{}]\tTotal testing loss: [{:.6f}]\tTotal testing error rate: [{:.6f}]'.format(
#			epoch, total_test_loss, (total_size - total_correct) / total_size * 100))
		meanTestLoss = float(total_test_loss)/len(self.test_loader)
		meanTestAcc = 1e2 * float(total_correct)/n_test
		print("Epoch:%d\t Mean test loss:%.6f,\tMean test acc:%.6f%%"%(
			epoch, meanTestLoss, meanTestAcc ) ) # n_train= number of total images in train set
		self.historyDict["meanTestLoss"][epoch] = meanTestLoss
		self.historyDict["meanTestAcc"][epoch] = meanTestAcc
		accuracy = meanTestAcc
		loss = meanTestLoss

		# writing logs into files
		info = {
			'epoch': epoch,
			'test-loss': loss,
			'test-top1-acc': accuracy
		}
		self.jsonlog_writer_test.write(json.dumps(info) + "\n")

#		if self.tflog_writer is not None:
#			info.pop('epoch', None)
#			for tag, value in info.items():
#				self.tflog_writer.scalar_summary(tag, value, )

		return accuracy, loss




	def retrieval_test(self, epoch):
		""" Testing epoch """
		print('==> Testing Epoch: %d' % epoch)
		self.net.eval()
		total_test_loss = 0
		total_correct = 0
		total_size = 0
		n_test = len(self.test_loader.dataset) # ->n_test

		val_dis = torch.tensor([]).cuda()
		val_flag = []
		for batch_idx, valid_pair in enumerate(self.trainloader_valid):
			pair_feature = []
			for pair_counter in range(2):
				inputs = valid_pair[pair_counter]
				img_input = torch.zeros(inputs.size(0), 3, 112, 96)
				for img_counter in range(inputs.size(0)):
					sampled = F.upsample(inputs[img_counter, :, :, :].view(1,3,224,224), 
										size=(112, 96), 
										mode='bilinear')
					img_input[img_counter, :,:,:] = sampled

				inputs = Variable(img_input.cuda(), volatile=True)
				with torch.no_grad():
					id_net_out = self.net(inputs)
				pair_feature.append(id_net_out)
			
			dis = self.cosine_dis(pair_feature[0], pair_feature[1])
			#print('\n    feature cmp:  ', pair_feature[0], '\n', pair_feature[1])
			val_dis = torch.cat((val_dis, dis), 0)
			val_flag = val_flag+valid_pair[2]
		print('\n\n\n\n\n', 'distance:', val_dis[:20], '\n\n\n\n\n')
		val_flag = torch.tensor(val_flag).cuda()
		pos_thresh = min(val_dis*val_flag)
		neg_thresh = max(val_dis*abs(val_flag-1))
		thresh = 0
		if pos_thresh <= neg_thresh:
			check_thresh = min(val_dis)
			check_step = (max(val_dis)-min(val_dis))/20
			acc_tmp = 0
			while check_thresh < max(val_dis):
				check_estimate = ( sum((val_dis<=check_thresh).float()*abs(check_thresh))+ \
								sum((val_dis>check_thresh).float()*val_flag) )/100
				check_estimate = float(check_estimate)
				if check_estimate > acc_tmp:
					acc_tmp = check_estimate
					thresh = check_thresh
				check_thresh += check_step
		else:
			acc_tmp = 1.0
			thresh = (neg_thresh+pos_thresh)/2

		print("Epoch:%d\t threshold:%.6f,\t instantaneous accuracy:%.6f%%"%(
			epoch, thresh, acc_tmp ) ) # n_train= number of total images in train set
		self.historyDict["meanTestLoss"][epoch] = thresh
		self.historyDict["meanTestAcc"][epoch] = acc_tmp
		accuracy = acc_tmp
		loss = float(thresh)
		# writing logs into files
		info = {
			'epoch': epoch,
			'test-loss': loss,
			'test-top1-acc': accuracy
		}
		self.jsonlog_writer_test.write(json.dumps(info) + "\n")
		return accuracy, loss



	def adjust_learning_rate(self, epoch):
		""" Sets the learning rate to the initial learning rate decayed by 10 every args.lr_decay_interval epochs """
		learning_rate = self.base_lr * (0.1 ** (epoch // self.lr_decay_interval))
		print('==> Set learning rate: %f' % learning_rate)
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = learning_rate

	def execute(self, end_epoch):
		for epoch in range(self.start_epoch, end_epoch):
			start = datetime.datetime.now()
			#self.adjust_learning_rate(epoch)
			self.train(epoch)
			accuracy, loss = self.retrieval_test(epoch)
			print('\n-------------------------retrieval test--------------------------------\n')
			print('accuracy:', accuracy)
			print('threshold:', loss)
			print('---------------------------------------------------------\n')
			accuracy, loss = self.test(epoch)

			# Save checkpoint.
			if accuracy > self.best_accuracy:
				print('==> Saving checkpoint..')
				self.best_accuracy = accuracy
				self.best_epoch = epoch
				state = {
					'start_epoch': epoch,
					'best_epoch': self.best_epoch,
					'best_accuracy': self.best_accuracy,
					'state_dict': self.net.state_dict(),
				}
				torch.save(state, osp.join(self.save_dir, 'ckpt.t7'))

			print('Best accuracy : %.6f%% from Epoch [%d]' % ( self.best_accuracy, self.best_epoch))
			torch.save(self.historyDict, self.historyDataFile)
			end = datetime.datetime.now()
			deltaSec = (end-start).total_seconds() # total_seconds return seonds. 1s=1e3 ms
			print("epoch {} takes {:.3f} seconds({:.1f} minutes+{:.3f} seconds).".format(epoch, deltaSec, deltaSec//60, deltaSec%60))
			strBuf = "epoch {}, test top1 acc: {:.7f}%\n".format(epoch, accuracy)
			os.system("echo -n \"{}\" >>checkpoint/validateLog.txt".format(strBuf))
