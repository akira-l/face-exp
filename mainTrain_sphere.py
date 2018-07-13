#!/usr/bin/env python3
# coding: utf-8

from __future__ import print_function
from __future__ import division

import os
import math
import os.path as osp
import argparse
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import torch.utils as utils
from torchvision import models, datasets, transforms
# from library.utils import ParameterReader
from library.classifyLoader import TraincsvSplit, ClassifyDataset
# transform is local foder, not standard pytorch transformation
#from transform.log_space import LogSpace
#from transform.disturb_illumination import DisturbIllumination

#datapath = "/home/storage/datasetAcademic/pipeProgram/PipeImg/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def calculate_mean_and_std(datapath,enable_log_transform):
	transform = transforms.Compose([
		transforms.Resize((224,224), interpolation=2),
		transforms.ToTensor(),
	])

	dataset = datasets.ImageFolder(root = datapath + 'train',transform = transform)
	dataloader = utils.data.DataLoader(dataset)
	data = np.stack([inputs[0].numpy() for inputs, targets in dataloader])
	mean = data.mean(axis=(0,2,3))
	std = data.std(axis=(0,2,3))
	return mean, std
def getMeanStdByBatch(datapath, _batch_size):
# @para: datapath, directory contain subdirectoried of different category images
# @para: _batch_size, how many images to fetch each time
# @return: totalMean, totalStd, mean and std(along the axis of RGB color) of all images in datapath.
	transform = transforms.Compose([
		transforms.Resize((224,224), interpolation=2),
		transforms.ToTensor(),
	])
	dataset = datasets.ImageFolder(root = datapath + 'train',transform = transform)
# dataset has totalLen(#img in directory) tuple, each tuple has 2 elements, first is img(chanel first order, then heighxwidth)
# second is label for that img
#>>> dataset.classes
#['left_abnormal', 'left_normal', 'mid_abnormal', 'mid_normal', 'right_abnormal', 'right_normal']
#>>> dataset.class_to_idx
#{'left_abnormal': 0, 'left_normal': 1, 'mid_abnormal': 2, 'mid_normal': 3, 'right_abnormal': 4, 'right_normal': 5}  
	dataloader = utils.data.DataLoader(dataset, batch_size = _batch_size)
# prepare mean & var: https://stackoverflow.com/questions/1480626/merging-two-statistical-result-sets
	totalLen = len(dataloader.dataset) # #img in directory
	nChannel = dataset[0][0].shape[0] #dataset[0]:tuple(img,label)
	totalMean = np.zeros( (nChannel) )
	totalVar = np.zeros( (nChannel) )
	for inputs, targets in dataloader:
		batchLen = len(inputs) # maybe improved
		batchMean = inputs.numpy().mean((0,2,3))
		batchVar = inputs.numpy().var((0,2,3))

		totalMean += batchMean * batchLen / totalLen
		totalVar += (batchVar+batchMean**2) * batchLen/totalLen
	totalVar -= totalMean**2
	totalStd = np.sqrt(totalVar)
	return totalMean, totalStd
if __name__ == '__main__':
	# Setup args
	parser = argparse.ArgumentParser(description='PyTorch SelfDataset Training')
	parser.add_argument('-lr','--learning-rate', type=float, default=0.001,
						help='initial learning rate (default: 0.001)')
	parser.add_argument('--epochs', type=int, default=100,
						help='number of epochs to train (default: 7)')
	parser.add_argument('--train-batch-size', type=int, default=16,
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=16,
						help='input batch size for testing (default: 64)')

	parser.add_argument('--lr-decay-interval', type=int, default=50,
						help='number of epochs to decay the learning rate (default: 50)')
	parser.add_argument('--num-workers', type=int, default=0,
						help='number of workers (default: 4)')
	parser.add_argument('--momentum', type=float, default=0.9,
						help='SGD momentum (default: 0.9)')
	parser.add_argument('--seed', type=int, default=1,
						help='random seed (default: 1)')

	parser.add_argument('-s', '--save-directory', type=str, default='checkpoint',
						help='checkpoint save directory (default: checkpoint)')
	parser.add_argument('-r', '--resume', action='store_true', default=False,
						help='resume from checkpoint')
	parser.add_argument('--paraFile', type=str, default="paraFile.txt",
						help='which parameter file you want to read from')

	args = parser.parse_args()

# 	paraReader = ParameterReader(args.paraFile)
	traincsvSplit = TraincsvSplit("../face_a","train.csv", 2874)
	classifyTrainCsv = "classifyTrain.csv"
	classifyValCsv = "classifyVal.csv"
	traincsvSplit.Split(classifyTrainCsv, classifyValCsv, 0.1)
	print('==> Init variables..')
	use_cuda = cuda.is_available()
	best_accuracy = 0  # best testing accuracy
	best_epoch = 0  # epoch with the best testing accuracy
	start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# 	netArch = "resnet34"
	netArch = "resnet50"
	save_directory = os.path.join("checkpoint", netArch)
	if not os.path.isdir(save_directory):
		os.makedirs(save_directory)

	# Init seed 
	print('==> Init seed..')
	torch.manual_seed(args.seed) # Sets the seed for generating random numbers
	if use_cuda:
		cuda.manual_seed(args.seed) # Sets the seed for generating random numbers for the current GPU

	# Calculate mean and std
	print('==> Prepare mean and std..')
#	data_mean, data_std = getMeanStdByBatch(datapath, args.train_batch_size)
# fengxi
	data_mean =  [ 0.331948,    0.33171957,  0.29903654]
	data_std =  [ 0.28179781,  0.27919075,  0.27801905]

	print('\tdata_mean = ', data_mean)
	print('\tdata_std = ', data_std)

	# Prepare training transform
	print('==> Prepare training transform..')
	training_transform = transforms.Compose([
# torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)
		torchvision.transforms.RandomAffine(0.5,translate=(0.003,0.003), scale=None, shear=0.5),
		transforms.Resize((224,224), interpolation=2),
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(data_mean, data_std)
	])
	print(training_transform)
	# Prepare testing transform
	print('==> Prepare testing transform..')

	testing_transform = transforms.Compose([
		transforms.Resize((224,224), interpolation=2),
		transforms.ToTensor(),
		transforms.Normalize(data_mean, data_std),
	])
	print(testing_transform)

	# Init 
	print('==> Init dataloader..')
	#
# 	trainset = datasets.ImageFolder(root = datapath + 'train',transform = training_transform)
	trainset = ClassifyDataset("../face_a/train/", classifyTrainCsv, training_transform)
	trainloader = utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

	#
# 	testset = datasets.ImageFolder(root = datapath + 'test',transform = testing_transform)
	testset = ClassifyDataset("../face_a/train/", classifyValCsv, testing_transform)
	testloader = utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

	# Model
	print('==> Building model..')
	# net = resnet_cifar.ResNet('res34', num_classes=100)
	kwargs = {"num_classes":2874}
	if netArch == "resnet18":
# 		net = torchvision.models.resnet.resnet18(**kwargs)
		net = torchvision.models.resnet.resnet18(pretrained=True)
	elif netArch == "resnet34":
# 		net = torchvision.models.resnet.resnet34(**kwargs)
		net = torchvision.models.resnet.resnet34(pretrained=True)
	elif netArch == "resnet50":
# 		net = torchvision.models.resnet.resnet50(**kwargs)
		net = torchvision.models.resnet.resnet50(pretrained=True)
	classifyInFeature = net.fc.in_features
	net.fc = nn.Linear(classifyInFeature, kwargs["num_classes"])
	if use_cuda:
		net = net.cuda()

	# Resume if required
	if args.resume:
		print('==> Resuming from checkpoint..')
		assert os.path.isdir(save_directory), 'Error: no checkpoint directory found!'
		if use_cuda:
			checkpoint = torch.load(save_directory + '/ckpt.t7')
		else:
			checkpoint = torch.load(save_directory + '/ckpt.t7', map_location=lambda storage, loc: storage)
		start_epoch = checkpoint['start_epoch']
		best_epoch = checkpoint['best_epoch']
		best_accuracy = checkpoint['best_accuracy']
		net.load_state_dict(checkpoint['state_dict'])

	# Loss function and Optimizer
	print('==> Setup loss function and optimizer..')
	criterion = nn.CrossEntropyLoss()
	if use_cuda:
		criterion = criterion.cuda()
	'''
	optimizer = optim.SGD(net.parameters(), lr=args.learning_rate,
						  momentum=args.momentum, weight_decay=1e-4,
						  nesterov=True)
	'''
# 	optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# 	optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), momentum=0.9,  eps=1e-08, weight_decay=0)
	optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
	# Training
	print('==> Init trainer..')
	from trainer import Trainer
	train = Trainer(net, trainloader, testloader, optimizer, start_epoch=start_epoch,
				 best_accuracy=best_accuracy, best_epoch=best_epoch, base_lr=args.learning_rate,
				 criterion=criterion, lr_decay_interval=args.lr_decay_interval, use_cuda=use_cuda, save_dir=save_directory,
				 totalEpoch=args.epochs, historyDataFile = os.path.join(save_directory, "historyData.t7"), outputInterval=20 )
	print('==> Start training..')
	train.execute(args.epochs)

