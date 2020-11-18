import numpy as np
import idx2numpy
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.utils.data 
from torch.autograd import Variable
import cv2
from sys import exit
import torch.nn as nn
import torch


class trainDataset( torch.utils.data.Dataset):
	def __init__(self, arr, lbls):
		self.arr = arr
		self.lbls = lbls

	def __len__(self):
		return self.arr.shape[0]
		
	def __getitem__(self, idx):
		img = self.arr[idx, :, :].reshape(1, 28, 28)
		lbl = self.lbls[idx]
		exit(1)

		img = torch.from_numpy(np.array(img)).float()
		# print(lbl.dtype); exit(1)

		return (img, lbl)


class testDataset( torch.utils.data.Dataset):
	def __init__(self, arr, lbls):
		self.arr = arr
		self.lbls = lbls

	def __len__(self):
		return self.arr.shape[0]
		
	def __getitem__(self, idx):
		img = self.arr[idx, :, :].reshape(1, 28, 28)
		lbl = self.lbls[idx]
		img = torch.from_numpy(np.array(img)).float()

		return (img, lbl)



def showImg(img):
	cv2.imshow("Image", img)
	cv2.waitKey(1000)



class fmnist(nn.Module):
	def __init__(self):
		super(fmnist, self).__init__()
		self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
		self.batch1=nn.BatchNorm2d(32)
		self.relu1 =nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2) 

		self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
		self.batch2=nn.BatchNorm2d(64)
		self.relu2 =nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2) 

		self.fc1= nn.Linear(64*5*5, 600)
		self.drop = nn.Dropout(0.25)
		self.fc2 = nn.Linear(in_features=600, out_features=120)
		self.fc3 = nn.Linear(in_features=120, out_features=10)

	def forward(self, x):
		 # Convolution 1
		out = self.cnn1(x)
		out = self.batch1(out)
		out = self.relu1(out)
		
		# Max pool 1
		out = self.maxpool1(out)
		
		# Convolution 2 
		out = self.cnn2(out)
		out = self.batch2(out)
		out = self.relu2(out)
		
		# Max pool 2 
		out = self.maxpool2(out)
		
		out = out.view(out.size(0), -1)

		# Linear function (readout)
		out = self.fc1(out)
		out= self.drop(out)
		out = self.fc2(out)
		out = self.fc3(out)

		return out



if __name__ == '__main__':
	trainImgsFile = 'fmnist/train-images-idx3-ubyte'
	trainImgs = idx2numpy.convert_from_file(trainImgsFile)

	trainLabelFile = 'fmnist/train-labels-idx1-ubyte'
	trainLbls = idx2numpy.convert_from_file(trainLabelFile)

	testImgsFile = 'fmnist/t10k-images-idx3-ubyte'
	testImgs = idx2numpy.convert_from_file(testImgsFile)

	testLabelFile = 'fmnist/t10k-labels-idx1-ubyte'
	testLbls = idx2numpy.convert_from_file(testLabelFile)

	epochs = 25
	batch_size = 8
	
	trainData = trainDataset(trainImgs, trainLbls)
	testData = testDataset(testImgs, testLbls)

	model = fmnist()

	criterion = nn.CrossEntropyLoss()
	learning_rate = 0.01

	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
	# for i in range(10):
	# 	img, lbl = trainData[i]

	# 	print(lbl)
	# 	showImg(img)

	train_loader = torch.utils.data.DataLoader(dataset=trainData, 
										   batch_size=batch_size, 
										   shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=testData, 
										  batch_size=batch_size, 
										  shuffle=False)
	for parm in range(len(list(model.parameters()))):
		 print(parm, "\t", list(model.parameters())[parm].size())
	
	for var_name in optimizer.state_dict():
		print(var_name, "\t", optimizer.state_dict()[var_name])

	torch.save(list(model.parameters()), 'ubyte.pth')	

	# model = fmnist(*args, **kwargs)
	# model.load_state_dict(torch.load('ubyte.pth'))
	# model.eval() 
	
	iter = 0
	for epoch in range(epochs):
		for i , (images , labels) in enumerate(train_loader):
			images = Variable(images)
			labels = Variable(labels)
			optimizer.zero_grad()
			
			# Forward pass to get output/logits
			outputs = model(images)
			
			# Calculate Loss: softmax --> cross entropy loss
			loss = criterion(outputs, labels.long())
			
			# Getting gradients w.r.t. parameters
			loss.backward()
			
			# Updating parameters
			optimizer.step()

			iter += 1
			
			if iter % 500 == 0:
				# Calculate Accuracy         
				correct = 0
				total = 0
				# Iterate through test dataset
				for images, labels in test_loader:
					images = Variable(images)

					outputs = model(images)
					
					# Get predictions from the maximum value
					_, predicted = torch.max(outputs.data, 1)
					
					# Total number of labels
					total += labels.size(0)
					
					correct += (predicted == labels).sum()
				
				accuracy = 100 * correct / total

				print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))