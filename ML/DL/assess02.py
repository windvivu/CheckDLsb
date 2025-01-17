import os
import sys
import torch
import torch.nn as nn

_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_dir)

from ML.DL._Dataset import SbDataset
from ML.DL._SimpleCNNsb import SimpleCNNsb, savecheckpoint

# Traning với model SimpleCNNsb.py
# Tính năng tuỳ chọn: có lấy lại file dataset đã lưu hay không (reuse_sbdtset = True), trong trường hợp muốn tạo lại dataset
# Không Training với weighted loss và weight sampler. Cứ để mất cân bằng dữ liệu xem sẽ như thế nào

if __name__ == "__main__":
	
	reuse_sbdtset = True
	
	batch_size = 32
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
	
	# trainsetsb.TURN2UP()
	# testsetsb.TURN2UP()

	if trainsetsb.turned != testsetsb.turned:
		raise ValueError("Trainset and testset must be turned the same way")
	
	train_dataloader = DataLoader(trainsetsb, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
	test_dataloader = DataLoader(testsetsb, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
	# checl old model file exist
	if os.path.exists(os.path.join(_dir, "_no_use/bestcheckpoint.chk")):
		checkpoint = torch.load(os.path.join(_dir, "_no_use/bestcheckpoint.chk"), weights_only=False)
		model = checkpoint["model"]
		bestaccu = checkpoint["accu"]
	else:
		if trainsetsb.turned == '':
			model = SimpleCNNsb().to(device)
		else:
			model = SimpleCNNsb(num_classes=2).to(device)
		bestaccu = 0

		
	criterion = nn.CrossEntropyLoss()

	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

	total_batch = len(train_dataloader)
	total_batch_test = len(test_dataloader)
	bestaccu = 0

	print('Begin training on:')
	trainsetsb.calculate_class_distribution(printOut=True)
	print('Class turning: ', trainsetsb.turned)

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
			info = {
				'ver': model.ver, 
				'indi': list2, 
				'bestacu': bestaccu,
				'classes': trainsetsb.class_to_idx2,
				'dtsturned': trainsetsb.turned
			}
			savecheckpoint(model, info, os.path.join(_dir, "_no_use/bestcheckpoint.chk"))
			with open(os.path.join(_dir, "_no_use/bestcheckpoint.txt"), "w") as f:
				f.write(str(bestaccu))