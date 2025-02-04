import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


class ImbalancedPrecisionFocusedLoss(nn.Module):
    def __init__(self, beta=2.0, class_weights=None, epsilon=1e-7):
        super().__init__()
        self.beta = beta  # beta > 1 để tập trung vào recall
        self.class_weights = class_weights
        self.epsilon = epsilon
        
    def forward(self, outputs, targets):
        # Áp dụng sigmoid cho binary classification
        probs = torch.sigmoid(outputs[:, 1])
        
        # Tính weights cho từng mẫu
        pos_weight = self.class_weights[1] if self.class_weights is not None else 3.0
        neg_weight = self.class_weights[0] if self.class_weights is not None else 1.0
        
        # Tính weighted BCE loss
        pos_loss = -targets * torch.log(probs + self.epsilon) * pos_weight
        neg_loss = -(1 - targets) * torch.log(1 - probs + self.epsilon) * neg_weight
        
        # Tính F-beta score focused loss
        TP = torch.sum(targets * probs)
        FP = torch.sum((1 - targets) * probs)
        FN = torch.sum(targets * (1 - probs))
        
        precision = TP / (TP + FP + self.epsilon)
        recall = TP / (TP + FN + self.epsilon)
        
        f_beta = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + self.epsilon)
        
        # Kết hợp BCE và F-beta loss
        bce_loss = torch.mean(pos_loss + neg_loss)
        f_beta_loss = 1 - f_beta
        
        return 0.7 * bce_loss + 0.3 * f_beta_loss


def find_optimal_threshold(y_true, y_prob, beta=2.0):
    best_threshold = 0.5
    best_f_beta = 0
    
    for threshold in torch.linspace(0.3, 0.7, 40):
        y_pred = (y_prob >= threshold).float()
        
        TP = torch.sum((y_pred == 1) & (y_true == 1)).float()
        FP = torch.sum((y_pred == 1) & (y_true == 0)).float()
        FN = torch.sum((y_pred == 0) & (y_true == 1)).float()
        
        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        
        f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-7)
        
        if f_beta > best_f_beta:
            best_f_beta = f_beta
            best_threshold = threshold
    
    return best_threshold

# def calculate_fbeta(y_true, y_prob, beta=2.0, threshold=None):
#     if threshold is None:
#         threshold = find_optimal_threshold(y_true, y_prob, beta)
    
#     y_pred = (y_prob >= threshold).float()
    
#     TP = torch.sum((y_pred == 1) & (y_true == 1)).float()
#     FP = torch.sum((y_pred == 1) & (y_true == 0)).float()
#     FN = torch.sum((y_pred == 0) & (y_true == 1)).float()
    
#     precision = TP / (TP + FP + 1e-7)
#     recall = TP / (TP + FN + 1e-7)
    
#     f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-7)
    
#     return f_beta, threshold

_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_dir)

from ML.DL._SimpleCNNsb2 import savecheckpoint, SimpleCNNsb, LightweightCNN, HybridCNN, EnhancedHybridCNN
from ML.DL._DepthWiseCNN import DepthWiseCNN

_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dir2 = os.path.dirname(_dir)
sys.path.append(_dir2)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

def savecheckpoint_textfile(model, info:dict, path_no_ext:str):
	savecheckpoint(model, info, path_no_ext + ".chk")
	with open(path_no_ext + ".txt", "w") as f:
		f.write('ver: ' + info["ver"] + '\n')
		f.write('turn: ' + info["dtsturned"] + '\n')
		f.write('accu: ' + str(info["accu"]) + '\n')
		f.write('f1: ' + str(info["f1"]) + '\n')
		f.write('loss: ' + str(info["loss"]) + '\n')
		f.write('epoch: ' + str(_epoch + epoch))

turndtset = 'up'
retrain = True
prefixchk = '_enh2'
saveby = 'loss' # accu, loss, f1
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
print('Load testsetsb.pth')
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


# Tăng trọng số cho lớp 1 thêm một chút #################
# class_weights[1] *= 1.2

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
if retrain and os.path.exists(os.path.join(_dir, f"_no_use/best{saveby}_checkpoint{turndtset}{prefixchk}.chk")): 
	checkpoint = torch.load(os.path.join(_dir, f"_no_use/best{saveby}_checkpoint{turndtset}{prefixchk}.chk"), weights_only=False, map_location=device)
	model = checkpoint["model"]
	bestaccu = checkpoint["info"]["accu"]
	bestloss = checkpoint["info"]["loss"]
	bestf1 = checkpoint["info"]["f1"]
	_epoch = checkpoint["info"]["epoch"]
	print(f'Re-train model from checkpoint: best{saveby}_checkpoint{turndtset}{prefixchk}')
else:
	if trainsetsb.turned == '':
		model = EnhancedHybridCNN().to(device)
	else:
		model = EnhancedHybridCNN(num_classes=2).to(device)
	bestaccu = 0
	bestloss = 100
	bestf1 = 0
	_epoch = 0
	
# ** Use weighted loss
# Without weights: Model might achieve 90% accuracy by just predicting majority class
# With weights: Model forced to learn minority classes better

# Thay thế dòng criterion cũ bằng:
criterion = ImbalancedPrecisionFocusedLoss(class_weights=class_weights).to(device)
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
	all_probs = []
	running_test_loss = 0.0

	for images, labels_test in tqdm(test_dataloader, desc=" Testing"):
		images = images.to(device)
		labels_test = labels_test.to(device)
		
		with torch.no_grad():
			outputs_test = model(images) # forward
		testloss = criterion(outputs_test, labels_test) # loss
		running_test_loss += testloss.item()  # Accumulate test loss

		probs = torch.sigmoid(outputs_test[:, 1])

		all_labels.extend(labels_test.cpu())
		all_probs.extend(probs.cpu())
	
	# Tìm ngưỡng tối ưu
	optimal_threshold = find_optimal_threshold(
    torch.tensor(all_labels), 
    torch.tensor(all_probs)
	)

	# Áp dụng ngưỡng tối ưu
	predictions = (torch.tensor(all_probs) >= optimal_threshold).float()
	all_predictions = predictions.tolist()
	
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
	print(f'Optimal Threshold: {optimal_threshold:.3f}')
	print(f'Accuracy: {accu:.4f}')
	print(f'Precision: {precision:.4f}')
	print(f'Recall: {recall:.4f}')
	print(f'F1-score: {f1:.4f}')
	print(f'Loss: {avg_test_loss:.4f}')

	conf_matrix = confusion_matrix(all_labels, all_predictions)
	print('\nConfusion Matrix:')
	print(conf_matrix)
	print(f'\nDetailed Metrics:')
	print(f'True Positives: {conf_matrix[1,1]}')
	print(f'False Positives: {conf_matrix[0,1]}')
	print(f'False Negatives: {conf_matrix[1,0]}')
	print(f'True Negatives: {conf_matrix[0,0]}')

	# Tính tỷ lệ phần trăm
	total_class_1 = conf_matrix[1,1] + conf_matrix[1,0]
	print(f'\nClass 1 Detection Rate: {conf_matrix[1,1]/total_class_1*100:.2f}%')
	print(f'False Positive Rate: {conf_matrix[0,1]/conf_matrix[0].sum()*100:.2f}%')

	if saveby == 'accu':
		if accu > bestaccu:
			bestaccu = accu
			info = {
				'ver': model.ver,  
				'accu': bestaccu,
				'f1': f1,
				'loss': avg_test_loss,
				'epoch': _epoch + epoch,
				'classes': trainsetsb.class_to_idx2,
				'dtsturned': trainsetsb.turned
			}
			savecheckpoint_textfile(model, info, os.path.join(_dir, f"_no_use/best{saveby}_checkpoint{turndtset}{prefixchk}"))
	elif saveby == 'loss':
		if avg_test_loss < bestloss:
			bestloss = avg_test_loss
			info = {
				'ver': model.ver, 
				'accu': accu,
				'f1': f1,
				'loss': bestloss,
				'epoch': _epoch + epoch,
				'classes': trainsetsb.class_to_idx2,
				'dtsturned': trainsetsb.turned
			}
			savecheckpoint_textfile(model, info, os.path.join(_dir, f"_no_use/best{saveby}_checkpoint{turndtset}{prefixchk}"))
	elif saveby == 'f1':
		if f1 > bestf1:
			bestf1 = f1
			info = {
				'ver': model.ver,  
				'accu': accu,
				'f1': bestf1,
				'loss': avg_test_loss,
				'epoch': _epoch + epoch,
				'classes': trainsetsb.class_to_idx2,
				'dtsturned': trainsetsb.turned
			}
			savecheckpoint_textfile(model, info, os.path.join(_dir, f"_no_use/best{saveby}_checkpoint{turndtset}{prefixchk}"))
	
	