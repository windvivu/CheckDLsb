import os
import pandas as pd
import time
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from signals._sigCore import sigcore
# from signals._classMytalib import addIndicatorList
# from ML._MakeSample import SampleXYtseries, addLabelUpDown
# from ML.RFRegressorClassifier import ModelRFRegressorClassifier
# import functions.getDTsqlite as mysqlite
import functions.myredis as rf

# https://ta-lib.github.io/ta-lib-python/

class SigExt(sigcore):
    def __init__(self, symbol, tf, market, testing:bool=False):
        '''
        https://ta-lib.github.io/ta-lib-python/
        
        '''
        super().__init__(symbol, tf, market, testing)
        self.name = os.path.basename(__file__).split('.py')[0]

        ## override
        self.datafolder = './'
        self.dt4train = None
        self.listI =   ['KAMA', 'MFI', 'DX', 'MA50', 'MINUS_DI', 'ULTOSC', 'ADX', 'DEMA', 'MA10']
        self.listI_2 = ['KAMA', 'MFI', 'AD', 'MA50', 'MINUS_DI', 'ULTOSC', 'ADX', 'ADXR', 'MA10', 'PLUS_DI', 'MIDPOINT', 'RSI']
        self.modelready = False


    def findsig(self, dfsub:pd.DataFrame, para:dict={}, hashkeystatus:str=''):
        import random
        ts = dfsub.iloc[-1]['timestamp']
        # cộng thêm 7 tiếng để đúng múi giờ GMT+7
        ts += 7*3600*1000
        dts = pd.to_datetime(ts, unit='ms').strftime('%Y-%m-%d %H:%M:%S')

        if self.modelready == False:
            # write log
            log = {'candle':f'{dts}', 'Content': f'[Findsig] skip for {dts}'}
            rf.writeLogModdetect(symbol=self.symbol, modname=self.name, log=log)
            return # chưa làm gì nếu model chưa sẵn sàng

        #---- thuật toán tìm sig/line_ và signal_/signallin_ ở đây
        dfsub['sig_'+self.name] = 0
        
        dfsub['signal_'+self.name] = 0

        random_value = random.choice([-1, 0, 1])
        dfsub.iloc[len(dfsub)-1, dfsub.columns.get_loc('signal_'+self.name)] = random_value
        if random_value == 0:
            log = {'candle':f'{dts}', 'Content': f'[Findsig] done(0/0) for {dts}'}
            rf.writeLogModdetect(symbol=self.symbol, modname=self.name, log=log)
        if random_value == 1:
            log = {'candle':f'{dts}', 'Content': f'[Findsig] done(1/0) for {dts}'}
            rf.writeLogModdetect(symbol=self.symbol, modname=self.name, log=log)
        if random_value == -1:
            log = {'candle':f'{dts}', 'Content': f'[Findsig] done(0/1) for {dts}'}
            rf.writeLogModdetect(symbol=self.symbol, modname=self.name, log=log)

        dfsub = dfsub.tail(1).copy()
        dfsub = dfsub.reset_index(drop=True)

        #---------------------------------------------------------
        dfsub = self._normalize_df(dfsub)
        self.dfsigs = dfsub[['timestamp', 'sig_'+self.name]]
        self.dfsignal = dfsub[['timestamp', 'signal_'+self.name]]

        if self.testing:
            return self.dfsigs, self.dfsignal, dfsub
        else:
            return self.dfsigs, self.dfsignal
        
    def firstInit(self, hashkeystatus:str=''):
        '''
        market: 'Binance' or 'FBinance'
        '''
        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Params init...')
        if self.market == 'Binance':
            self.datafolder = '/data/BNspot'
        elif self.market == 'FBinance':
            self.datafolder = '/data/BNfuture'
        else:
            self.datafolder = '/data/YFstocks'
        # pastCandle=14, foreCast=5, thresoldDiff=0.02,
        self.pastCandle = 14
        self.foreCast = 5
        self.thresoldDiff = 0.02 

        self.limit = self.foreCast + 2 # số lượng dòng tín hiệu sẽ lưu vào redis. Nên lớn hơn forecast ít nhất 1, như thế sig sẽ không bị trống do chưa đủ nến tương lai.
    
    # def secondInit(self, hashkeystatus:str=''):
    #     pass
    
    def thirdJob(self, hashkeystatus:str='', setDelay:bool=True):
        # chuyển công việc training model cho thirdJob, tức là sau khi tìm ra tín hiệu. Như vậy sẽ giúp tìm tín hiệu lần sau nhanh hơn.
        self._trainmodels(hashkeystatus)
        self.modelready = True

        # write log
        log = {'Content': '[ThirdJob] done'}
        rf.writeLogModdetect(symbol=self.symbol, modname=self.name, log=log)

    def _trainmodels(self, hashkeystatus:str=''):
        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Model training...')
        for i in range(3):
            time.sleep(1)
        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Models ready...')
