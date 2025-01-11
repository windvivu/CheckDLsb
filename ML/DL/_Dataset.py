import os
import sys
import torch
from torch.utils.data import Dataset
class SbDataset(Dataset):
	def __init__(self, root, datafolder, symbol, timeframe, listIndi, train=True, typeMake=3, testsize=1000, label='close', pastCandle=15, foreCast=5, thresoldDiff=0.02):
		sys.path.append(root)
		import functions.getDTsqlite as mysqlite
		from signals._classMytalib import addIndicatorList
		from ML._MakeSample import SampleXYtseries

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

		self.Y = None

		X, Y = self._proccessdt(symbol, timeframe)
				
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

	
	def showplotly(self, index):
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
		if self.typeMake == 2:
			return {0: 0, 1: 1}
		elif self.typeMake == -2:
			return {0: 0, -1: 1}
		else:
			return {0: 0, -1: 1, 1: 2}
	
	@property
	def class_to_idx2(self):
		if self.typeMake == 2:
			return {'None': 0, 'Up': 1}
		elif self.typeMake == -2:
			return {'None': 0, 'Down': 1}
		else:
			return {'None': 0, 'Down': 1, 'Up': 2}

import torch.nn as nn

class SimpleCNNsb(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.ver = 'sb0'

        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
        )

		# Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
        )

		# Third conv block
        self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Dropout2d(0.4),
			# nn.MaxPool2d(2),
		)

		# Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


    def forward(self,x):
       x = self.conv1(x)
       x = self.conv2(x)
       x = self.conv3(x)
       x = x.view(x.size(0), -1)  # Flatten
       x = self.fc(x)
       return x

# save model
def savecheckpoint(model, ver, indi, bestacu, filename):
    checkpoint = {
      "model": model,
      "ver": ver,
	  "indi": indi,
	  "accu": bestacu
	}
    torch.save(checkpoint, filename)

if __name__ == "__main__":
	
	reuse_sbdtset = True
	
	batch_size = 16
	num_epochs = 100000
	
	_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	_dir2 = os.path.dirname(_dir)
	sys.path.append(_dir2)
	from torch.utils.data import DataLoader, WeightedRandomSampler
	from tqdm import tqdm
	# check device
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")


	list2 = ["MA10", "DEMA", "EMA26", "KAMA", "MIDPRICE",   "ADX", "ADXR", "DX", "MFI", "MINUS_DI", "PLUS_DI", "RSI", "ULTOSC","WILLR",   "NATR", "CMO"]
	
	# kiểm tra xem "trainsetsb.pth" đã tồn tại chưa, sau đo save hoặc load file với torch
	if os.path.exists(os.path.join(_dir, "_no_use/trainsetsb.pth")) and reuse_sbdtset:
		trainsetsb = torch.load(os.path.join(_dir, "_no_use/trainsetsb.pth"), weights_only=False)
	else:
		# tạo trainsetsb	
		trainsetsb = SbDataset(root=_dir2, datafolder="data/BNfuture", symbol="NEO/USDT", timeframe="4h", listIndi=list2)
		# save trainsetsb to file by torch
		torch.save(trainsetsb, os.path.join(_dir, "_no_use/trainsetsb.pth"))

	# kiểm tra xem "testsetsb.pth" đã tồn tại chưa, sau đo save hoặc load file với torch
	if os.path.exists(os.path.join(_dir, "_no_use/testsetsb.pth")) and reuse_sbdtset:
		testsetsb = torch.load(os.path.join(_dir, "_no_use/testsetsb.pth"), weights_only=False)
	else:
		# tạo testsetsb	
		testsetsb = SbDataset(root=_dir2, datafolder="data/BNfuture", symbol="NEO/USDT", timeframe="4h", listIndi=list2, train=False)
		# save testsetsb to file by torch
		torch.save(testsetsb, os.path.join(_dir, "_no_use/testsetsb.pth"))
	
	trainsetsb.ADDMORETRAINDT("BNB/USDT", "4h")
	torch.save(trainsetsb, os.path.join(_dir, "_no_use/trainsetsb_more.pth"))

	trainsetsb.calculate_class_distribution(printOut=True)
	exit()

	class_weights = trainsetsb.calculate_class_distribution(printOut=False).to(device)
	weights = []
	for label in trainsetsb.targets:
		weights.append(class_weights[label])
	# ** Use WeightedRandomSampler
	sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
    replacement=True
	)
	# with sampler,  shuffle must be False
	train_dataloader = DataLoader(trainsetsb, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0, drop_last=False)
	test_dataloader = DataLoader(testsetsb, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
	# checl old model file exist
	if os.path.exists(os.path.join(_dir, "_no_use/bestcheckpoint.chk")):
		checkpoint = torch.load(os.path.join(_dir, "_no_use/bestcheckpoint.chk"), weights_only=False)
		model = checkpoint["model"]
		bestaccu = checkpoint["accu"]
	else:
		model = SimpleCNNsb().to(device)
		bestaccu = 0

		
	# ** Use weighted loss
	# Without weights: Model might achieve 90% accuracy by just predicting majority class
	# With weights: Model forced to learn minority classes better
	criterion = nn.CrossEntropyLoss(weight=class_weights)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

	total_batch = len(train_dataloader)
	total_batch_test = len(test_dataloader)
	bestaccu = 0

	for epoch in range(num_epochs):
		print("Epoch: ", str(epoch+1), "/", str(num_epochs))
		model.train()
		progress_bar = tqdm(train_dataloader, desc=" Training", colour="green")
		for iter, (images, labels_train) in enumerate(progress_bar):
			images = images.to(device)
			labels_train = labels_train.to(device)
			
			outputs_train = model(images) # forward
			loss = criterion(outputs_train, labels_train) # loss
			
			optimizer.zero_grad() # gradient -> làm sach buffer gradient
			loss.backward() # backward
			optimizer.step() # update weight
		
		print(' End epoch', epoch+1,'loss value of training:', loss.item())
		model.eval()
		all_predictions = []
		all_labels = []
		for images, labels_test in tqdm(test_dataloader, desc=" Testing"):
			images = images.to(device)
			labels_test = labels_test.to(device)
			
			all_labels.extend(labels_test)
			with torch.no_grad():
				outputs_test = model(images) # forward
				max_to_label = torch.argmax(outputs_test, dim=1)
                #loss = criterion(outputs_test, labels_test) # loss
				all_predictions.extend(max_to_label)
		print(' End epoch', epoch+1, end=' - ')
		accu = (torch.tensor(all_predictions) == torch.tensor(all_labels)).sum().item()/len(all_labels)
		print('Accuracy:', accu)
		
		if accu > bestaccu:
			bestaccu = accu
			savecheckpoint(model, model.ver, list2, bestaccu, os.path.join(_dir, "_no_use/bestcheckpoint.chk"))
			with open(os.path.join(_dir, "_no_use/bestcheckpoint.txt"), "w") as f:
				f.write(str(bestaccu))
		
		# savecheckpoint(model, os.path.join(savepath, "lastcheckpoint.pt"))
		# with open(os.path.join(savepath, "lastmodel.txt"), "w") as f:
		# 	f.write(str(accu))

	
	
