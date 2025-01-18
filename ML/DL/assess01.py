import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_dir)

from ML.DL._Dataset import SbDataset
from ML.DL._SimpleCNNsb import savecheckpoint, SimpleCNNsb, SimpleCNNsbkernel5x5, SimpleCNNsbkernel7x7, SimpleCNNsbkernel75

# Traning với model SimpleCNNsb.py
# Tính năng tuỳ chọn: có lấy lại file dataset đã lưu hay không (reuse_sbdtset = True), trong trường hợp muốn tạo lại dataset
# Training với weighted loss và weight sampler để giảm thiểu hiện tượng mất cân bằng dữ liệu

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
        
#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
#         pt = torch.exp(-ce_loss)
#         focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         return focal_loss

if __name__ == "__main__":
	
	reuse_sbdtset = True
	test_phase = False
	turndtset = 'none'
	if test_phase:
		sysboltest = "NEO/USDT"
		tftest = "4h"
	
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


	list2 = ["MA10", "DEMA", "EMA26", "KAMA", "MIDPRICE", "close",   "ADX", "ADXR", "DX", "MFI", "MINUS_DI", "PLUS_DI", "RSI", "ULTOSC","WILLR",   "NATR"] # thay CMO thành close, do thấy CMO chẳng khác gì RSI
	# list2 = ["MFI", "RSI", "close"]
	
	# kiểm tra xem "trainsetsb.pth" đã tồn tại chưa, sau đo save hoặc load file với torch
	if os.path.exists(os.path.join(_dir, "_no_use/trainsetsb.pth")) and reuse_sbdtset:
		trainsetsb = torch.load(os.path.join(_dir, "_no_use/trainsetsb.pth"), weights_only=False)
	else:
		# tạo trainsetsb	
		trainsetsb = SbDataset(root=_dir2, datafolder="data/BNfuture", symbol="NEO/USDT", timeframe="4h", listIndi=list2)
		# save trainsetsb to file by torch
		torch.save(trainsetsb, os.path.join(_dir, "_no_use/trainsetsb.pth"))

	testsetsb = None
	if test_phase:
		# kiểm tra xem "testsetsb.pth" đã tồn tại chưa, sau đo save hoặc load file với torch
		if os.path.exists(os.path.join(_dir, "_no_use/testsetsb.pth")) and reuse_sbdtset:
			testsetsb = torch.load(os.path.join(_dir, "_no_use/testsetsb.pth"), weights_only=False)
		else:
			# tạo testsetsb	
			testsetsb = SbDataset(root=_dir2, datafolder="data/BNfuture", symbol=sysboltest, timeframe=tftest, listIndi=list2, train=False)
			# save testsetsb to file by torch
			torch.save(testsetsb, os.path.join(_dir, "_no_use/testsetsb.pth"))
	
	if turndtset == 'up':
		trainsetsb.TURN2UP()
		if testsetsb is not None: testsetsb.TURN2UP()
	elif turndtset == 'down':
		trainsetsb.TURN2DOWN()
		if testsetsb is not None: testsetsb.TURN2DOWN()
	elif turndtset == 'none':
		trainsetsb.TURN2NONE()
		if testsetsb is not None: testsetsb.TURN2NONE()
	
	if test_phase:
		if trainsetsb.turned != testsetsb.turned:
			raise ValueError("Trainset and testset must be turned the same way")

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
	if test_phase: test_dataloader = DataLoader(testsetsb, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
	
	# checl old model file exist
	checkpoint = None
	if os.path.exists(os.path.join(_dir, "_no_use/bestcheckpoint" + turndtset + ".chk")):
		checkpoint = torch.load(os.path.join(_dir, "_no_use/bestcheckpoint" + turndtset + ".chk"), weights_only=False)
		model = checkpoint["model"]
		bestloss = checkpoint["loss"]
		_epoch = checkpoint["info"]["epoch"]
	else:
		if trainsetsb.turned == '':
			model = SimpleCNNsb().to(device)
		else:
			model = SimpleCNNsb(num_classes=2).to(device)
		# bestaccu = 0
		bestloss = 100
		_epoch = 0

	v = 'sb0' # make sure same version of model loaded from file: sb0, sb1k55, sb1k77, sb1k75
	if checkpoint is not None:
		if checkpoint['info']['ver'] != v:
			print("Wrong version of model")
			exit()

		
	# ** Use weighted loss
	# Without weights: Model might achieve 90% accuracy by just predicting majority class
	# With weights: Model forced to learn minority classes better
	
	criterion = nn.CrossEntropyLoss(weight=class_weights)
	# criterion = FocalLoss(alpha=class_weights, gamma=2.0)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)


	print('Begin training', end='')
	if test_phase:
		print(' and testing...')
	else:
		print(' only...')
	trainsetsb.calculate_class_distribution(printOut=True)
	print('Class turning: [', trainsetsb.turned,']')

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
		
		lossItem = loss.item()
		print(' End epoch', epoch+1,'loss value of training:', lossItem)
		
		if test_phase:
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
			accu = (torch.tensor(all_predictions) == torch.tensor(all_labels)).sum().item()/len(all_labels)
			print('Accuracy:', accu)
			print(' End epoch', epoch+1, end=' - ')
		
		if lossItem < bestloss:
			bestloss = lossItem
			info = {
				'ver': model.ver, 
				'indi': list2, 
				'bestloss': bestloss,
				'epoch': _epoch + epoch,
				'classes': trainsetsb.class_to_idx2,
				'dtsturned': trainsetsb.turned
			}
			savecheckpoint(model, info, os.path.join(_dir, "_no_use/bestcheckpoint" + turndtset + ".chk"))
			with open(os.path.join(_dir, "_no_use/bestcheckpoint" + turndtset + ".txt"), "w") as f:
				f.write(model.ver + '\n')
				f.write( trainsetsb.turned + '\n')
				f.write(str(bestloss) + '\n')
				f.write(str(_epoch + epoch))