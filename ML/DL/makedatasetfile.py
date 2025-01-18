import os
import sys
import torch

reuse_sbdtset = False
make_testset = False

list2 = ["MA10", "DEMA", "EMA26", "KAMA", "MIDPRICE", "close",   "ADX", "ADXR", "DX", "MFI", "MINUS_DI", "PLUS_DI", "RSI", "ULTOSC","WILLR",   "NATR"] # thay CMO thành close, do thấy CMO chẳng khác gì RSI
# list2 = ["MFI", "RSI", "close"]

_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dir2 = os.path.dirname(_dir)
sys.path.append(_dir2)
from ML.DL._Dataset import SbDataset

print('Convert data to trainset....')
# kiểm tra xem "trainsetsb.pth" đã tồn tại chưa, sau đo save hoặc load file với torch
if os.path.exists(os.path.join(_dir, "_no_use/trainsetsb.pth")) and reuse_sbdtset:
	print("Reuse trainset file...")
	trainsetsb = torch.load(os.path.join(_dir, "_no_use/trainsetsb.pth"), weights_only=False)
else:
	# tạo trainsetsb
	if make_testset:
		testsize = 1000
	else:
		testsize = 0
	print("Making initial trainset...")
	symbol="NEO/USDT"
	tf = "4h"	
	trainsetsb = SbDataset(root=_dir2, datafolder="data/BNfuture", symbol=symbol, timeframe=tf, listIndi=list2, testsize=testsize)
	# save trainsetsb to file by torch
	torch.save(trainsetsb, os.path.join(_dir, "_no_use/trainsetsb.pth"))
	print("Done with NEO/USDT")

# kiểm tra xem "testsetsb.pth" đã tồn tại chưa, sau đo save hoặc load file với torch
if make_testset:
	if os.path.exists(os.path.join(_dir, "_no_use/testsetsb.pth")) and reuse_sbdtset:
		testsetsb = torch.load(os.path.join(_dir, "_no_use/testsetsb.pth"), weights_only=False)
	else:
		# tạo testsetsb	
		testsetsb = SbDataset(root=_dir2, datafolder="data/BNfuture", symbol="QTUM/USDT", timeframe="4h", listIndi=list2, train=False)
		# save testsetsb to file by torch
		torch.save(testsetsb, os.path.join(_dir, "_no_use/testsetsb.pth"))

	# exit()

print('Add more data to trainset 1...')
symbol="BNB/USDT"
trainsetsb.ADDMORETRAINDT(symbol, tf)
torch.save(trainsetsb, os.path.join(_dir, "_no_use/trainsetsb.pth"))
print('Done with', symbol, tf)

print('Add more data to trainset 2...')
symbol="ETH/USDT"
trainsetsb.ADDMORETRAINDT(symbol, tf)
torch.save(trainsetsb, os.path.join(_dir, "_no_use/trainsetsb.pth"))
print('Done with', symbol, tf)