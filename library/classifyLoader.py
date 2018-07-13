import os
from PIL import Image
import csv
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
class TraincsvSplit(object):
	def __init__(self, inputFile, numOfClass):
		self.inputFile = inputFile
		self.totalList = list(csv.reader(open(inputFile, "r") ) )
		self.numOfClass = numOfClass
	def Split(self, trainFile, valFile, valRatio):
		length = len(self.totalList)
		memberDict = {i:[] for i in range(self.numOfClass)}
		trainList = []
		valList = []
		for i in range(length):
			id = self.totalList[i] [1]
			memberDict[int(id)].append(i) # list of integer
		valsetNum = int(length * valRatio)
		for i in range(len(memberDict)):
			if len(memberDict[i])>1:
				if len(valList)<valsetNum:
					valList.append(self.totalList[ memberDict[i][0]  ])
				else:
					trainList.append(self.totalList[ memberDict[i][0]  ])
				for j in range(1,len(memberDict[i])):
					trainList.append(self.totalList[ memberDict[i][j]  ])
			elif len(memberDict[i]) == 1:
				trainList.append(self.totalList[ memberDict[i][0]  ])
		with open(trainFile,"w") as trainFP:
			for line in trainList:
				strBuf = ','.join(line)+'\n'
				trainFP.write(strBuf)
		with open(valFile, "w") as valFP:
			for line in valList:
				strBuf = ','.join(line)+'\n'
				valFP.write(strBuf)
class ClassifyDataset(data.Dataset):
	def __init__(self, root, csvFile, dataTransform):
		self.root = root
		self.csvFile = csvFile
		self.dataTransform = dataTransform
		file_list = csv.reader(open(self.csvFile,'r'))
		file_list = list(file_list)
		for i in range(len(file_list)):
			file_list[i][0] = os.path.join(self.root, file_list[i][0] )
			file_list[i][1] = int(file_list[i][1])
		self.file_list = file_list

	def __getitem__(self, idx):
		intLabel = self.file_list[idx][1]
		imgFileName = self.file_list[idx][0]
		img = Image.open(imgFileName)		
		img = self.dataTransform(img)
		return (img, intLabel, imgFileName)
	def __len__(self):
		return len(self.file_list)
