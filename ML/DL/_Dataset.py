import os
import sys
import torch
from torch.utils.data import Dataset
class SbDataset(Dataset):
	def __init__(self, root, datafolder, symbol, timeframe, listIndi, train=True, typeMake=3, testsize=1000, label='close', pastCandle=15, foreCast=5, thresoldDiff=0.02):
		sys.path.append(root)

		self.root = root
		self.datafolder = datafolder
		self.symbol = symbol
		self.timeframe = timeframe
		self.listIndi = listIndi
		self.train = train
		self.typeMake = typeMake

		self.label = label
		self.pastCandle = pastCandle
		self.foreCast = foreCast
		self.thresoldDiff = thresoldDiff

		self.spl = None

		self.Y = None
		self.Y_temp = None
		self.turned = ''	

		X, Y = self._proccessdt(symbol, timeframe)

		self.initialLenTrainset = len(X[:-testsize])
				
		if train:
			self.X = X[:-testsize]
			self.Y = Y[:-testsize]

			# trộn ngẫu nhiên dữ liệu
			perm = torch.randperm(len(self.X))
			self.X = self.X[perm]
			self.Y = self.Y[perm]
		else:
			self.X = X[-testsize:]
			self.Y = Y[-testsize:]

		# conver numpy X, y to tensor
		self.X = torch.tensor(self.X, dtype=torch.float32)
		
		# Thêm một chiều mới ở vị trí thứ 1
		self.X = self.X.unsqueeze(1)

		self.Y = torch.tensor(self.Y, dtype=torch.int)	
		
	def _proccessdt(self, symbol, timeframe):
		sys.path.append(self.root)
		import functions.getDTsqlite as mysqlite
		from signals._classMytalib import addIndicatorList
		from ML._MakeSample import SampleXYtseries

		df = mysqlite.getDTsqlite(os.path.join(self.root, self.datafolder), symbol=symbol, timeframe=timeframe)
		df = addIndicatorList(df, self.listIndi)
		self.spl = SampleXYtseries(data=df, typeMake=self.typeMake, features=self.listIndi, label=self.label, pastCandle=self.pastCandle, foreCast=self.foreCast, thresoldDiff=self.thresoldDiff, typeScale='minmax')
		print("Processing data to matrix...")
		X, Y = self.spl.PROCESSMATRIX(realDT=False, showProgress=True, square=True)

		return X, Y
	
	def _turn2class(self, type:str):
		'''
		type: up or down or none or ''
		'''
		if self.Y is None:
			raise ValueError("You must create train dataset first!")
		if self.typeMake != 3:
			raise ValueError("Cannot do with initial typemake < 3")
		
		if type == 'up':
			if self.turned != '':
				raise ValueError("You must turn back to '' before turn to 'up'")
			
			# save self.Y to self.Y_temp
			self.Y_temp = self.Y
			# if item in self.Y > 1, set to 1, else set to 0 
			self.Y = torch.where(self.Y == 2, 1, 0)
			self.turned = 'up'
		elif type == 'down':
			if self.turned != '':
				raise ValueError("You must turn back to '' before turn to 'down'")
			
			# save self.Y to self.Y_temp
			self.Y_temp = self.Y
			# if item in self.Y = 1, set to 1, else set to 0 
			self.Y = torch.where(self.Y == 1, 1, 0)
			self.turned = 'down'
		elif type == 'none':
			if self.turned != '':
				raise ValueError("You must turn back to '' before turn to 'none'")
			
			# save self.Y to self.Y_temp
			self.Y_temp = self.Y
			# if item in self.Y = 0, set to 1, else set to 0
			self.Y = torch.where(self.Y == 0, 1, 0)
			self.turned = 'none'
		elif type == '':
			if self.turned in ('up', 'down', 'none'):
				self.Y = self.Y_temp
				self.turned = ''
		else:
			raise ValueError("type must be 'up' or 'down' or none or ''")

	
	def __getitem__(self, index):
		return self.X[index], self.Y[index].item()

	def __len__(self):
		return len(self.X)
	
	def ADDMORETRAINDT(self, symbol, timeframe):
		if self.Y is None:
			raise ValueError("You must create train dataset first!")
		
		X, Y = self._proccessdt(symbol, timeframe)
		X = torch.tensor(X, dtype=torch.float32)
		X = X.unsqueeze(1)
		Y = torch.tensor(Y, dtype=torch.int)

		# add to current traine dataset
		self.X = torch.cat((self.X, X), 0)
		self.Y = torch.cat((self.Y, Y), 0)

	def TURN2UP(self):
		self._turn2class('up')
	
	def TURN2DOWN(self):
		self._turn2class('down')
	
	def TURN2NONE(self):
		self._turn2class('none')
	
	def TURN3CLASS(self):
		self._turn2class('')
	
	def showplotly(self, index):
		if self.train:
			raise ValueError("Cannot show plot for trainset because this set was suffled")
		else:
			index = self.initialLenTrainset + index
			self.spl.plotlySampleX(index)

	def calculate_class_distribution(self, printOut=True):
		labels = self.Y
		unique_labels, counts = torch.unique(labels, return_counts=True)
		percentages = counts.float() / len(self.Y) * 100
		# Print distribution
		if printOut:
			print("\nClass Distribution:")
			for label, count, percent in zip(unique_labels, counts, percentages):
				print(f"Class {label}: {count} samples ({percent:.2f}%)")
		
		class_weights = len(labels) / (len(unique_labels) * counts.float())
		return class_weights
	
	@property
	def targets(self):
		return self.Y
	
	@property
	def classes(self):
		return self.Y.unique().tolist()
	
	@property
	def class_to_idx(self):
		if self.typeMake == 2 or self.turned == 'up':
			return {0: 0, 1: 1}
		elif self.typeMake == -2 or self.turned == 'down':
			return {0: 0, -1: 1}
		elif self.typeMake == 3 and self.turned == 'none':
			return {0.5: 0, 0: 1}
		else:
			return {0: 0, -1: 1, 1: 2}
	
	@property
	def class_to_idx2(self):
		if self.typeMake == 2 or self.turned == 'up':
			return {'None & down': 0, 'Up': 1}
		elif self.typeMake == -2 or self.turned == 'down':
			return {'None & up': 0, 'Down': 1}
		elif self.typeMake == 3 and self.turned == 'none':
			return {'Down & up': 0, 'None': 1}
		else:
			return {'None': 0, 'Down': 1, 'Up': 2}

