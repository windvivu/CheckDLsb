import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import pandas as pd


_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dir2 = os.path.dirname(_dir)
sys.path.append(_dir2)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

pathCheckpoint = "_no_use/bestcheckpoint009.chk"  #####
pathCheckpoint_v = "_no_use/bestcheckpoint012.chk"  #####


# check device
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

# kiểm tra xem "testsetsb.pth" đã tồn tại chưa, sau đo save hoặc load file với torch
if os.path.exists(os.path.join(_dir, "_no_use/testsetsb.pth")):
	testsetsb = torch.load(os.path.join(_dir, "_no_use/testsetsb.pth"), weights_only=False)
else:
	print("No testset file found")
	exit()

# checl old model file exist
if os.path.exists(os.path.join(_dir, pathCheckpoint)):
	checkpoint = torch.load(os.path.join(_dir, pathCheckpoint), 
					  map_location=device,
					  weights_only=False)
	model = checkpoint["model"]
else:
	print("No model file found")
	exit()

# check old model v file exist
if os.path.exists(os.path.join(_dir, pathCheckpoint_v)):
	checkpoint_v = torch.load(os.path.join(_dir, pathCheckpoint_v), 
					  map_location=device,
					  weights_only=False)
	model_v = checkpoint_v["model"]
else:
	print("No model_v file found")
	exit()

print("Model for predict: ", model.ver)
print("Model for verify: ", model_v.ver)

print("Model turned: (", checkpoint['info']['dtsturned'], ")")
print("Model_v turned: (", checkpoint_v['info']['dtsturned'], ")")

try:
	if checkpoint['info']['dtsturned'] == 'up':
		testsetsb.TURN2UP()
	elif checkpoint['info']['dtsturned'] == 'down':
		testsetsb.TURN2DOWN()
except Exception as e: # ngoại lệ cho checkpoint cũ không có thông tin về turned
	# testsetsb.TURN2UP()
	raise ValueError("Old checkpoint file must have turned information")


nums = len(testsetsb)
print('Number of sample: ', nums)
# duyệt qua từng phần tử trong testset và dự đoán
# ghi kết quả vào một pandas dataframe
tbl = {
	'Label': [],
	'Predict': [],
}

def doPredict(model, sample):
	model.eval()
	with torch.no_grad():
		outputs_test = model(sample)
	
	max_to_label = torch.argmax(outputs_test, dim=1)
	pred = max_to_label.item()
	
	return pred

for i in tqdm(range(nums)):
	sample, lb = testsetsb[i]
	tbl['Label'].append(lb)
	
	# _sample = sample.clone()
	sample = sample.unsqueeze(0).to(device)
	# _sample = _sample.unsqueeze(0).to(device)
	# _sample[:, :, 15, :] = 0  # -> 3.1

	# predict with model
	pred = doPredict(model, sample)

	# if pred == 1:
	# 	# verify with model_v
	# 	pred = doPredict(model_v, sample)
		# if pred == 0:
		# 	pred = 1
		# else:
		# 	pred = 0
		
	tbl['Predict'].append(pred)
	

# tbl = pd.DataFrame(tbl)
# tbl.to_excel('predict_result_' + checkpoint['info']['dtsturned'] + '.xlsx', index=False)
print(confusion_matrix(tbl['Label'], tbl['Predict']))
print(classification_report(tbl['Label'], tbl['Predict']))


