import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_dir)

from ML.DL._Dataset import SbDataset
from ML.DL._SimpleCNNsb import savecheckpoint, SimpleCNNsb, SimpleCNNsbkernel5x5, SimpleCNNsbkernel7x7, SimpleCNNsbkernel75

# Traning với model SimpleCNNsb.py
# Tính năng tuỳ chọn: có lấy lại file dataset đã lưu hay không (reuse_sbdtset = True), trong trường hợp muốn tạo lại dataset
# Training với weighted loss và weight sampler để giảm thiểu hiện tượng mất cân bằng dữ liệu


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        """
        Args:
            patience (int): Số epoch chờ đợi trước khi dừng training
            min_delta (float): Ngưỡng cải thiện tối thiểu
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # Validation loss không giảm đủ nhiều
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            # Validation loss giảm đủ nhiều
            self.best_loss = val_loss
            self.counter = 0
        return False

def savecheckpoint_textfile(model, info:dict, path_no_ext:str):
	savecheckpoint(model, info, path_no_ext + ".chk")
	with open(path_no_ext + ".txt", "w") as f:
		f.write('ver: ' + info["ver"] + '\n')
		f.write('turn: ' + info["dtsturned"] + '\n')
		f.write('accu: ' + str(info["accu"]) + '\n')
		f.write('f1: ' + str(info["f1"]) + '\n')
		f.write('loss: ' + str(info["loss"]) + '\n')
		f.write('epoch: ' + str(_epoch + epoch))
	
reuse_sbdtset = True
saveby = 'loss' # accu, loss, f1
useEarlyStop = False
useScheduler = False
turndtset = 'down'
sysboltest = "NEO/USDT"
tftest = "4h"

batch_size = 32
num_epochs = 100000
list2 = ["MA10", "DEMA", "EMA26", "KAMA", "MIDPRICE", "close",   "ADX", "ADXR", "DX", "MFI", "MINUS_DI", "PLUS_DI", "RSI", "ULTOSC","WILLR",   "NATR"] # thay CMO thành close, do thấy CMO chẳng khác gì RSI
# list2 = ["MFI", "RSI", "close"]

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

# kiểm tra xem "trainsetsb.pth" đã tồn tại chưa
if os.path.exists(os.path.join(_dir, "_no_use/trainsetsb.pth")) and reuse_sbdtset:
	trainsetsb = torch.load(os.path.join(_dir, "_no_use/trainsetsb.pth"), weights_only=False)
else:
	print('Err: No trainset file found!')
	exit()


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
test_dataloader = DataLoader(testsetsb, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

# checl old model file exist
checkpoint = None
if os.path.exists(os.path.join(_dir, "_no_use/bestcheckpoint010.chk")): # os.path.join(_dir, f"_no_use/best{saveby}_checkpoint{turndtset}.chk"
	checkpoint = torch.load(os.path.join(_dir, f"_no_use/bestcheckpoint010.chk"), weights_only=False, map_location=device)
	model = checkpoint["model"]
	bestaccu = 0#checkpoint["info"]["accu"]
	bestloss = 0#checkpoint["info"]["loss"]
	bestf1 = 0#checkpoint["info"]["f1"]
	_epoch = checkpoint["info"]["epoch"]
else:
	if trainsetsb.turned == '':
		model = SimpleCNNsb().to(device)
	else:
		model = SimpleCNNsb(num_classes=2).to(device)
	bestaccu = 0
	bestloss = 100
	bestf1 = 0
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',          # giảm learning rate khi metric không giảm
    patience=3,          # đợi 3 epoch
    factor=0.1,         # giảm learning rate 10 lần
    min_lr=1e-6,        # learning rate tối thiểu
    verbose=True        # in thông báo khi learning rate thay đổi
)

early_stopping = EarlyStopping()

print('Begin training...')

trainsetsb.calculate_class_distribution(printOut=True)
print('Class turning: [', trainsetsb.turned,']')
for epoch in range(num_epochs):
	print("Epoch: ", str(epoch+1), "/", str(num_epochs))
	model.train()
	progress_bar = tqdm(train_dataloader, desc=" Training", colour="green")
	running_loss = 0.0
	for iter, (images, labels_train) in enumerate(progress_bar):
		images = images.to(device)
		labels_train = labels_train.to(device)
		
		outputs_train = model(images) # forward
		trainloss = criterion(outputs_train, labels_train) # loss
		
		optimizer.zero_grad() # gradient -> làm sach buffer gradient
		trainloss.backward() # backward
		optimizer.step() # update weight

		running_loss += trainloss.item()
		break
	
	avg_loss = running_loss / len(progress_bar)
	print(' End epoch', epoch+1,'loss value of training:', avg_loss)
	
	model.eval()
	all_predictions = []
	all_labels = []
	running_test_loss = 0.0  # Initialize test loss

	for images, labels_test in tqdm(test_dataloader, desc=" Testing"):
		images = images.to(device)
		labels_test = labels_test.to(device)
		
		all_labels.extend(labels_test)
		with torch.no_grad():
			outputs_test = model(images) # forward
		max_to_label = torch.argmax(outputs_test, dim=1)
		testloss = criterion(outputs_test, labels_test) # loss
		running_test_loss += testloss.item()  # Accumulate test loss
		all_predictions.extend(max_to_label)
	
	# Calculate average test loss
	avg_test_loss = running_test_loss / len(test_dataloader)

	# Convert lists to tensors once
	pred_tensor = torch.tensor(all_predictions).to(device)
	label_tensor = torch.tensor(all_labels).to(device)
	
	# Calculate accuracy
	correct = (pred_tensor == label_tensor).sum().item()
	total = len(all_labels)
	accu = correct / total

	# Calculate metrics
	pred_cpu = pred_tensor.cpu().numpy()
	label_cpu = label_tensor.cpu().numpy()
	if trainsetsb.turned == '':
		precision, recall, f1, _ = precision_recall_fscore_support(label_cpu, pred_cpu, average='weighted')
	else:
		precision, recall, f1, _ = precision_recall_fscore_support(label_cpu, pred_cpu, average='binary')

	if useScheduler: scheduler.step(avg_test_loss) # Use average test loss
	current_lr = optimizer.param_groups[0]['lr']
	print(f'Current learning rate: {current_lr}')
	print(f'Test metrics:')
	print(f'- Accuracy: {accu:.4f}')
	print(f'- Precision: {precision:.4f}')
	print(f'- Recall: {recall:.4f}') 
	print(f'- F1-score: {f1:.4f}')
	print(f'- Loss value of testing: {avg_test_loss}')
	print('== End epoch', epoch+1, end=' - ')
	
	if saveby == 'accu':
		if accu > bestaccu:
			bestaccu = accu
			info = {
				'ver': model.ver, 
				'indi': list2, 
				'accu': bestaccu,
				'f1': f1,
				'loss': avg_test_loss,
				'epoch': _epoch + epoch,
				'classes': trainsetsb.class_to_idx2,
				'dtsturned': trainsetsb.turned
			}
			savecheckpoint_textfile(model, info, os.path.join(_dir, f"_no_use/best{saveby}_checkpoint{turndtset}"))
	elif saveby == 'loss':
		if avg_test_loss < bestloss:
			bestloss = avg_test_loss
			info = {
				'ver': model.ver, 
				'indi': list2, 
				'accu': accu,
				'f1': f1,
				'loss': bestloss,
				'epoch': _epoch + epoch,
				'classes': trainsetsb.class_to_idx2,
				'dtsturned': trainsetsb.turned
			}
			savecheckpoint_textfile(model, info, os.path.join(_dir, f"_no_use/best{saveby}_checkpoint{turndtset}"))
	elif saveby == 'f1':
		if f1 > bestf1:
			bestf1 = f1
			info = {
				'ver': model.ver, 
				'indi': list2, 
				'accu': accu,
				'f1': bestf1,
				'loss': avg_test_loss,
				'epoch': _epoch + epoch,
				'classes': trainsetsb.class_to_idx2,
				'dtsturned': trainsetsb.turned
			}
			savecheckpoint_textfile(model, info, os.path.join(_dir, f"_no_use/best{saveby}_checkpoint{turndtset}"))
			
	
	if useEarlyStop and early_stopping(avg_test_loss):
		print("Early stopping !!!!!!!!!!!!!")
		break