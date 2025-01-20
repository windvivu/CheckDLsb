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

pathCheckpoint_up = "_no_use/bestcheckpoint009.chk"  #####
pathCheckpoint_down = "_no_use/bestcheckpoint010.chk"  #####
pathCheckpoint_none = "_no_use/bestcheckpoint000.chk"  #####


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
if os.path.exists(os.path.join(_dir, pathCheckpoint_up)):
	checkpoint_up = torch.load(os.path.join(_dir, pathCheckpoint_up), 
					  map_location=device,
					  weights_only=False)
	model_up = checkpoint_up["model"]
else:
	print("No model_up file found")
	exit()

# check old model v file exist
if os.path.exists(os.path.join(_dir, pathCheckpoint_down)):
	checkpoint_down = torch.load(os.path.join(_dir, pathCheckpoint_down), 
					  map_location=device,
					  weights_only=False)
	model_down = checkpoint_down["model"]
else:
	print("No model_down file found")
	exit()

# check old model none file exist
if os.path.exists(os.path.join(_dir, pathCheckpoint_none)):
	checkpoint_none = torch.load(os.path.join(_dir, pathCheckpoint_none), 
					  map_location=device,
					  weights_only=False)
	model_none = checkpoint_none["model"]
else:
	print("No model_none file found")
	exit()

print("Model for predict: ", model_up.ver)
print("Model for verify: ", model_down.ver)
print("Model for verify 2: ", model_none.ver)

print("Model turned: (", checkpoint_up['info']['dtsturned'], ")")
print("Model_v turned: (", checkpoint_down['info']['dtsturned'], ")")
print("Model_v_none turned: (", checkpoint_none['info']['dtsturned'], ")")

# try:
# 	if checkpoint_up['info']['dtsturned'] == 'up':
# 		testsetsb.TURN2UP()
# 	elif checkpoint_down['info']['dtsturned'] == 'down':
# 		testsetsb.TURN2DOWN()
# except Exception as e: # ngoại lệ cho checkpoint cũ không có thông tin về turned
# 	# testsetsb.TURN2UP()
# 	raise ValueError("Old checkpoint file must have turned information")
# testsetsb.TURN2UP()

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

def doPredictUp(model_up, sample, threshold=1.1):
	score = 0
	score_d = 0
	pred = doPredict(model_up, sample)
	if pred == 1: # tức là up  (nhưng gồm up, down, none), precision chỉ 30%
		score += 1
	else: # tức down và none, precision 70%
		pass
		score_d += 1

	pred_v = doPredict(model_down, sample)
	if pred_v == 0: # tức là up or none (precision cao)
		score += 0.1
		score_d -= 0.1
	else: # tức là down
		pass # không tác động vì precision thấp
		# score -= 0.1
	
	pred_v_none = doPredict(model_none, sample)
	if pred_v_none == 0: # tức up and down
		pass
		score += 0.1
		score_d += 0.1
	else:
		# pass
		score -= 0.1 # tức là none
		score_d -= 0.1
	
	if score >= threshold:
		pred = 2
	else:
		pred = 0
	
	# print('pred ', pred, 'score ', score, 'score_d', score_d )

	# if pred ==0 and score_d >= 0.2:
	# 	pred = 1
	return pred

def doPredictDown(model_down, sample, threshold=1.1):
	score = 0
	pred = doPredict(model_down, sample)
	if pred == 1: # tức là down  (nhưng gồm up, down, none), precision chỉ 30%
		score += 1
	else: # tức up và none, precision 70%
		pass
	pred_v = doPredict(model_up, sample)
	if pred_v == 0: # tức là down or none (precision cao)
		score += 0.1
	else: # tức là up
		pass # không tác động vì precision thấp
		# score -= 0.1
	
	pred_v_none = doPredict(model_none, sample)
	if pred_v_none == 0: # tức up and down
		pass
		score += 0.1
	else:
		# pass
		score -= 0.1 # tức là none
	
	if score >= threshold:
		pred = 1
	else:
		pred = 0

	return pred

for i in range(nums):
	sample, lb = testsetsb[i]
	tbl['Label'].append(lb)
	
	sample = sample.unsqueeze(0).to(device)

	predict = doPredictUp(model_up, sample)
	# if predict == 1: predict = 2

		
	tbl['Predict'].append(predict)
	

# tbl = pd.DataFrame(tbl)
# tbl.to_excel('predict_result_' + checkpoint['info']['dtsturned'] + '.xlsx', index=False)
print(confusion_matrix(tbl['Label'], tbl['Predict']))
print(classification_report(tbl['Label'], tbl['Predict'], zero_division=0))