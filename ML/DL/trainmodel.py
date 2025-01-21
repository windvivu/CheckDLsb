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
_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dir2 = os.path.dirname(_dir)
sys.path.append(_dir2)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

turndtset = 'up'
batch_size = 32
num_epochs = 100000

# check device
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

# kiểm tra xem "trainsetsb.pth" đã tồn tại chưa
print('Load trainsetsb.pth')
if os.path.exists(os.path.join(_dir, "_no_use/trainsetsb.pth")):
	trainsetsb = torch.load(os.path.join(_dir, "_no_use/trainsetsb.pth"), weights_only=False)
else:
	print('Err: No trainset file found!')
	exit()

# kiểm tra xem "testsetsb.pth" đã tồn tại chưa
print('testsetsb.pth')
if os.path.exists(os.path.join(_dir, "_no_use/testsetsb.pth")) :
	testsetsb = torch.load(os.path.join(_dir, "_no_use/testsetsb.pth"), weights_only=False)
else:
	print("Err: No testset found!")
	exit()

if turndtset == 'up':
	trainsetsb.TURN2UP()
	testsetsb.TURN2UP()
elif turndtset == 'down':
	trainsetsb.TURN2DOWN()
	testsetsb.TURN2DOWN()
elif turndtset == 'none':
	trainsetsb.TURN2NONE()
	testsetsb.TURN2NONE()

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

if trainsetsb.turned == '':
	model = SimpleCNNsb().to(device)
else:
	model = SimpleCNNsb(num_classes=2).to(device)
	
	
# ** Use weighted loss
# Without weights: Model might achieve 90% accuracy by just predicting majority class
# With weights: Model forced to learn minority classes better

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


print('Begin training...')

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
	
	avg_loss = running_loss / len(progress_bar)
	print(' End training', epoch+1,'loss value of training:', avg_loss)
	
	model.eval()
	all_predictions = []
	all_labels = []
	running_test_loss = 0.0  # Initialize test loss
	for images, labels_test in tqdm(test_dataloader, desc=" Testing"):
		images = images.to(device)
		labels_test = labels_test.to(device)
		
		with torch.no_grad():
			outputs_test = model(images) # forward
		testloss = criterion(outputs_test, labels_test) # loss
		running_test_loss += testloss.item()  # Accumulate test loss

		max_to_label = torch.argmax(outputs_test, dim=1)
		all_labels.extend(labels_test)
		all_predictions.extend(max_to_label)
	
	# Convert lists to tensors once
	label_tensor = torch.tensor(all_labels).to(device)
	pred_tensor = torch.tensor(all_predictions).to(device)

	# Calculate accuracy
	correct = (pred_tensor == label_tensor).sum().item()
	total = len(all_labels)
	accu = correct / total

	# Calculate metrics
	label_numpy = label_tensor.cpu().numpy()
	pred_numpy = pred_tensor.cpu().numpy()
	if trainsetsb.turned == '':
		precision, recall, f1, _ = precision_recall_fscore_support(label_numpy, pred_numpy, average='weighted')
	else:
		precision, recall, f1, _ = precision_recall_fscore_support(label_numpy, pred_numpy, average='binary')


	# Calculate average test loss
	avg_test_loss = running_test_loss / len(test_dataloader)

	print('Test-------------------------------------')
	print(f'- Accuracy: {accu}')
	print(f'- Precision: {precision}')
	print(f'- Recall: {recall}') 
	print(f'- F1-score: {f1}')
	print(f'- Loss value of testing: {avg_test_loss}')
	print('== End epoch', epoch+1, end=' - ')
	
	