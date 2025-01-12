import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_dir)

from ML.DL._Dataset import SbDataset
from ML.DL._SimpleCNNsb import SimpleCNNsb

if __name__ == "__main__":
	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['OPENBLAS_NUM_THREADS'] = '1'	
	
	batch_size = 32

	pathCheckpoint = "_no_use/bestcheckpoint008.chk"  #####

	_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	_dir2 = os.path.dirname(_dir)
	sys.path.append(_dir2)
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
		bestaccu = checkpoint["accu"]
	else:
		print("No model file found")
		exit()
	
	try:
		if checkpoint['info']['dtsturned'] == 'up':
			testsetsb.TURN2UP()
		elif checkpoint['info']['dtsturned'] == 'down':
			testsetsb.TURN2DOWN()
	except: # ngoại lệ cho checkpoint cũ không có thông tin về turned
		pass
		testsetsb.TURN2UP()

	test_dataloader = DataLoader(testsetsb, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

	print("Begin prediction with model accu", model.ver, bestaccu)
	print(checkpoint['info'])
	model.eval()
	all_predictions = []
	all_labels = []
	for images, labels_test in tqdm(test_dataloader, desc=" Testing"):
		images = images.to(device)
		labels_test = labels_test.to(device)
		
		all_labels.extend(labels_test)
		with torch.no_grad():
			outputs_test = model(images)
			max_to_label = torch.argmax(outputs_test, dim=1)
			all_predictions.extend(max_to_label)
	
	accu = (torch.tensor(all_predictions) == torch.tensor(all_labels)).sum().item()/len(all_labels)
	print('Accuracy:', accu)
	print("Confusion matrix:")
	print(confusion_matrix(all_labels, all_predictions))
	print("Classification report:")
	print(classification_report(all_labels, all_predictions))

	# exit()

	# print('---------------------Test 1 saple----------------------------------------------------------------------')

	# test 1 sample
	# index_sample = 12
	# a, b = testsetsb[index_sample]
	# testsetsb.showplotly(index_sample)