import os
import sys
import torch
from torch.utils.data import Dataset
class SbDataset(Dataset):
	def __init__(self, root, datafolder, symbol, timeframe, listIndi, train=True, typeMake=3, testsize=1000):
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

		df = mysqlite.getDTsqlite(os.path.join(root, datafolder), symbol=symbol, timeframe=timeframe)
		df = addIndicatorList(df, listIndi)
		self.spl = SampleXYtseries(data=df, typeMake=typeMake, features=listIndi, label='close', pastCandle=15, foreCast=5, thresoldDiff=0.02, typeScale='minmax')
		print("Processing data to matrix...")
		X, Y = self.spl.PROCESSMATRIX(realDT=False, showProgress=True, square=True)
		
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
		
	def __getitem__(self, index):
		return self.X[index], self.Y[index].item()

	def __len__(self):
		return len(self.X)
	
	def showplotly(self, index):
		self.spl.plotlySampleX(index)
	
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

class RefDataset(Dataset):
	def __init__(self, root, train=True):
		from torchvision.transforms import ToTensor, Resize, Compose
		import torchvision.datasets as dts
		
		self.train = train
		
		transform = Compose([Resize((32,32)), ToTensor()])
		self.dataset = dts.CIFAR10(root=root, train=train, download=True, transform=transform)
	
	def __getitem__(self, index):
		# return imagetensor, label 
		# De thuc hien resize, totensor, convert("RGB"). Conver RGB này để tránh gặp
		# ảnh có 4 kênh màu (1 kênh trong suốt) thay vì 3 kênh
		# from torchvision.transforms import ToTensor, Resize
		# img = self.transform(img) ; transform =Compose([Resize((x,y)), ToTensor()])
		#  ảnh sẽ có dạng (3,x,y)
		# không tính các bộ nhỏ như CIFA thì ảnh chuẩn hiện nay thường là 3x224x224
		# lý do là 224 chia hết 32. Bộ nhỏ CIFA kích thước 3x32x32
		return self.dataset[index] 
	
	def __len__(self):
		return len(self.dataset)
	
	@property
	def classes(self):
		return self.dataset.classes
	
	@property
	def class_to_idx(self):
		return self.dataset.class_to_idx

import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3, out_channels=8)
        self.conv2 = self.make_block(in_channels=8, out_channels=16)
        self.conv3 = self.make_block(in_channels=16, out_channels=32)
        self.conv4 = self.make_block(in_channels=32, out_channels=64)
        self.conv5 = self.make_block(in_channels=64, out_channels=128)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=512),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes),
        )

    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.shape[0], -1) # flatten 4d to 2d

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
	
class SimpleCNNsb(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = self.make_block(in_channels=1, out_channels=8)
        self.conv2 = self.make_block(in_channels=8, out_channels=16)
        self.conv3 = self.make_block(in_channels=16, out_channels=32)
        self.conv4 = self.make_block(in_channels=32, out_channels=64)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=512),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes),
        )

    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.shape[0], -1) # flatten 4d to 2d

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
	
	type_run = 1 # 0: image, 1: sb  #################
	
	batch_size = 16
	num_epochs = 100
	
	_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	_dir2 = os.path.dirname(_dir)
	sys.path.append(_dir2)
	from torch.utils.data import DataLoader
	from tqdm import tqdm
	# check device
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	if type_run == 1:	
		list2 = ["KAMA", "DEMA", "ADXR", "ULTOSC", "NATR","MFI", "EMA26", "ADX", "MINUS_DI", "WILLR", "DX", "MIDPRICE", "PLUS_DI", "CMO", "RSI"]

		# kiểm tra xem "trainsetsb.pth" đã tồn tại chưa, sau đo save hoặc load file với torch
		if os.path.exists(os.path.join(_dir, "_no_use/trainsetsb.pth")):
			trainsetsb = torch.load(os.path.join(_dir, "_no_use/trainsetsb.pth"))
		else:
			# tạo trainsetsb	
			trainsetsb = SbDataset(root=_dir2, datafolder="data/BNfuture", symbol="NEO/USDT", timeframe="4h", listIndi=list2)
			# save trainsetsb to file by torch
			torch.save(trainsetsb, os.path.join(_dir, "_no_use/trainsetsb.pth"))
	
		# kiểm tra xem "testsetsb.pth" đã tồn tại chưa, sau đo save hoặc load file với torch
		if os.path.exists(os.path.join(_dir, "_no_use/testsetsb.pth")):
			testsetsb = torch.load(os.path.join(_dir, "_no_use/testsetsb.pth"))
		else:
			# tạo testsetsb	
			testsetsb = SbDataset(root=_dir2, datafolder="data/BNfuture", symbol="NEO/USDT", timeframe="4h", listIndi=list2, train=False)
			# save testsetsb to file by torch
			torch.save(testsetsb, os.path.join(_dir, "_no_use/testsetsb.pth"))
		
		train_dataloader = DataLoader(trainsetsb, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
		test_dataloader = DataLoader(testsetsb, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

		model = SimpleCNNsb().to(device)

	else:
		
		trainset = RefDataset(root=os.path.join(_dir, "_no_use"))
		testset = RefDataset(root=os.path.join(_dir, "_no_use"), train=False)

		train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
		test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

		model = SimpleCNN().to(device)


	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
			# savecheckpoint(model, os.path.join(savepath, "bestcheckpoint.pt"))
			# with open(os.path.join(savepath, "bestmodel.txt"), "w") as f:
			# 	f.write(str(bestaccu))
		# savecheckpoint(model, os.path.join(savepath, "lastcheckpoint.pt"))
		# with open(os.path.join(savepath, "lastmodel.txt"), "w") as f:
		# 	f.write(str(accu))

	
	
