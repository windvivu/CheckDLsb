import os
import pandas as pd
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from signals._sigCore import sigcore
from signals._classMytalib import addIndicatorList
from ML._MakeSample import SampleXYtseries, addLabelUpDown
from ML.RFRegressorClassifier import ModelRFRegressorClassifier
import functions.getDTsqlite as mysqlite
import functions.f4run as f4run
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
        if rf.check_hash('info_' + symbol):
            self.testmode = int(rf.get_hash('info_' + symbol, 'testmode'))
            self.GMTs = int(rf.get_hash('info_' + symbol, 'GMTs'))
            self.gapcandle = int(rf.get_hash('info_' + symbol, 'gapCandle'))
        else:
            self.testmode = 1
            self.GMTs = 7
            self.gapcandle = None
        if self.testing:
            self.testmode = 1

    def findsig(self, dfsub:pd.DataFrame, para:dict={}, hashkeystatus:str=''):
        ts = dfsub.iloc[-1]['timestamp']
        ts += self.GMTs*3600*1000
        dts = pd.to_datetime(ts, unit='ms').strftime('%Y-%m-%d %H:%M:%S')

        # self._trainmodels() # cho train luôn để test  #temporary
        # self.modelready = True   #temporary

        if self.modelready == False: 
            # write log
            log = {'candle':f'{dts}','Content': f'[Findsig] skip for {dts}'}
            rf.writeLogModdetect(symbol=self.symbol, modname=self.name, log=log)
            return # chưa làm gì nếu model chưa sẵn sàng

        #---- thuật toán tìm sig/line_ và signal_/signallin_ ở đây
        dfsub2 = addIndicatorList(dfsub, self.listI)
        dfsub_2 = addIndicatorList(dfsub, self.listI_2)

        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'UPredicting...')
        y2 = self.md2.ProcessPredict(dfsub2, realDT=True, copyScalerIfPredict=True) # copyScalerIfPredict ngầm định là True. Tuy nhiên viết ra ở đây để thấy tầm quan trọng của nó
        
        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'DPredicting...')
        y_2 = self.md_2.ProcessPredict(dfsub_2, realDT=True, copyScalerIfPredict=True) # copyScalerIfPredict ngầm định là True. Tuy nhiên viết ra ở đây để thấy tầm quan trọng của nó

        sig_ = addLabelUpDown(dfsub, self.foreCast, self.thresoldDiff) 
        sig_ = sig_* 100  
        dfsub['sig_'+self.name] = sig_        

        minlen = min(len(dfsub), len(y2), len(y_2))
        
        dfsub = dfsub[len(dfsub)-minlen:].copy()
        dfsub = dfsub.reset_index(drop=True)

        y2 = y2[len(y2)-minlen:].copy()
        y_2 = y_2[len(y_2)-minlen:].copy()

        dfsub['y2'] = pd.Series(y2)
        dfsub['y_2'] = pd.Series(y_2)
        
        dfsub['signal_'+self.name] = 0
        writtenlog = False
        for i in range(len(dfsub)):
            if dfsub.iloc[i, dfsub.columns.get_loc('y2')] == 1 and dfsub.iloc[i, dfsub.columns.get_loc('y_2')] == 0:
                dfsub.iloc[i, dfsub.columns.get_loc('signal_'+self.name)] = 1
            if dfsub.iloc[i, dfsub.columns.get_loc('y2')] == 0 and dfsub.iloc[i, dfsub.columns.get_loc('y_2')] == 1:
                dfsub.iloc[i, dfsub.columns.get_loc('signal_'+self.name)] = -1
            
            if self.testmode == 0:
                if i == len(dfsub)-2: # lưu ý là len-2 vì chỉ ghi log lấy dòng áp cuối. Bước sau sẽ shift(1) để gán giá trị dòng cuối bằng dòng áp cuối.
                    log = {'candle':f'{dts}', 'Content': f'[Findsig] done({dfsub.iloc[i, dfsub.columns.get_loc('y2')]}{dfsub.iloc[i, dfsub.columns.get_loc('y_2')]}) for {dts}'}
                    rf.writeLogModdetect(symbol=self.symbol, modname=self.name, log=log)
                    writtenlog = True
            else:
                if i == len(dfsub)-1:
                    log = {'candle':f'{dts}', 'Content': f'[Findsig] done({dfsub.iloc[i, dfsub.columns.get_loc('y2')]}{dfsub.iloc[i, dfsub.columns.get_loc('y_2')]}) for {dts}'}
                    rf.writeLogModdetect(symbol=self.symbol, modname=self.name, log=log)
                    writtenlog = True
        
        # chuyển gia trị cột sig, signal dịch xuống 1 hàng. Lý do là khi chạy realtime nến cuối 
        # cùng vẫn chưa có giá close cố định nên không thể dự đoán chính xác. Còn nếu chạy thử nghiệm
        # thì không cần dịch vì dữ liệu nến cuối là dữ liệu đã đóng nến.       
        
        # dfsub.to_excel('dfsub__' + str(ts) + '.xlsx') #temporary 
                
        if self.testmode == 0:
            dfsub['sig_'+self.name] = dfsub['sig_'+self.name].shift(1)
            dfsub['signal_'+self.name] = dfsub['signal_'+self.name].shift(1)
            dfsub.dropna(inplace=True)
        
        # dfsub.to_excel('dfsub' + str(ts) + 'shift1.xlsx') #temporary

        # write log
        if writtenlog == False:
            log = {'candle':f'{dts}', 'Content': f'[Findsig] done for {dts}'}
            rf.writeLogModdetect(symbol=self.symbol, modname=self.name, log=log)

        #---------------------------------------------------------
        dfsub = self._normalize_df(dfsub)
        self.dfsigs = dfsub[['timestamp', 'sig_'+self.name]]
        self.dfsignal = dfsub[['timestamp', 'signal_'+self.name]]

        if self.testing:
            return self.dfsigs, self.dfsignal, dfsub
        else:
            return self.dfsigs, self.dfsignal
    
    #---- hàm ghi vào redis----override hàm này thay vì dùng hàm kế thừa do có các yêu cầu đặc biệt
    def save_signal_to_redis(self):
        '''
        Ghi tín hiệu vào redis
        '''
        if self.dfsigs.empty or self.dfsignal.empty:
            return None

        # nếu đã tồn tại hash sig thì chỉ ghi giới hạn số dòng mới
        if rf.check_exist_df_sig(self.symbol, self.name):
            # ghi lại đường, dấu hiệu, model có giá trị để thể hiện trên chart
            rf.add_df_sig(self.dfsigs.tail(self.limit), self.symbol, self.name)
            # ghi lại tín hiệu xuất hiện vị thế
            rf.add_df_signal(self.dfsignal.tail(1), self.symbol, self.name)
        # chưa tồn tại thì ghi toàn bộ nhưng RIÊNG SIGNAL VẪN CHỈ GHI 1 DÒNG CUỐI CÙNG
        else:
            rf.add_df_sig(self.dfsigs, self.symbol, self.name)
            rf.add_df_signal(self.dfsignal.tail(1), self.symbol, self.name)

        # get value in column signal of last row
        return self.dfsignal.iloc[-1]['signal_'+self.name]
        
    def firstInit(self, hashkeystatus:str=''):
        '''
        market: 'Binance' or 'FBinance'
        '''
        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Params init...')
        if self.market == 0: # Binance
            self.datafolder = '/data/BNspot'
        elif self.market == 1: # FBinance
            self.datafolder = '/data/BNfuture'
        else:
            self.datafolder = '/data/YFstocks'
        # pastCandle=14, foreCast=5, thresoldDiff=0.02,
        self.pastCandle = 14
        self.foreCast = 5
        self.thresoldDiff = 0.02 

        self.limit = self.foreCast*2 # số lượng dòng tín hiệu sẽ lưu vào redis. Nên lớn hơn forecast như thế sig sẽ không bị trống do chưa đủ nến tương lai.
    
    def secondInit(self, hashkeystatus:str=''):
        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Second init...')
    
    def thirdJob(self, hashkeystatus:str='', setDelay:bool=True):
        # chuyển công việc training model cho thirdJob, tức là sau khi tìm ra tín hiệu. Như vậy sẽ giúp tìm tín hiệu lần sau nhanh hơn.
        # trong chạy realtime, nếu timeframe từ 1h trở lên thì nên delay một khoảng ngẫu nhiên để tránh training cùng lúc nhiều model
        if self.testmode == 0 and setDelay:
            if self.gapcandle >= 60*60*1000:
                import random
                import time
                delay = random.randint(1, int(self.gapcandle*(2/3))//1000)
                for i in range(delay):
                    if rf.check_hash('info_' + self.symbol) == False: return
                    if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Wait for ' + str(delay - i) + 's...')
                    time.sleep(1)

        # training model
        self._trainmodels(hashkeystatus)
        self.modelready = True

        # write log
        log = {'Content': '[ThirdJob] done'}
        rf.writeLogModdetect(symbol=self.symbol, modname=self.name, log=log)

    def _trainmodels(self, hashkeystatus:str=''):

        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Get sample data') 
        self.dt4train = mysqlite.getDTsqlite(_parent_dir + self.datafolder, f4run.cSymbol(self.symbol), self.tf, silent=True)
        
        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Add indi for updt')
        self.dt4trainIndi2 = addIndicatorList(self.dt4train, self.listI)

        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Add indi for downdt')
        self.dt4trainIndi_2 = addIndicatorList(self.dt4train, self.listI_2)

        self.spl2 = SampleXYtseries(self.dt4trainIndi2, typeMake=2, features=self.listI, typeScale='minmax', pastCandle=self.pastCandle, foreCast=self.foreCast, thresoldDiff=self.thresoldDiff)
        self.md2 = ModelRFRegressorClassifier(self.spl2, class_weight='balanced', n_estimators=300, max_depth=None, min_samples_split = 2, min_samples_leaf = 1, max_features = 'log2', internalInfo=False)

        self.spl_2 = SampleXYtseries(self.dt4trainIndi_2, typeMake=-2, features=self.listI_2, typeScale='minmax', pastCandle=self.pastCandle, foreCast=self.foreCast, thresoldDiff=self.thresoldDiff)
        self.md_2 = ModelRFRegressorClassifier(self.spl_2, class_weight='balanced', n_estimators=300, max_depth=None, min_samples_split = 2, min_samples_leaf = 1, max_features = 'log2', internalInfo=False)

        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Udata processing...')
        self.md2.ProcessDataPCAhardsplit(suffle=True)
        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Ddata processing...')
        self.md_2.ProcessDataPCAhardsplit(suffle=True)

        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Umodel training...')
        self.md2.TrainHardModel()
        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Dmodel training...')
        self.md_2.TrainHardModel()
        if hashkeystatus != '': rf.add_key_value(hashkeystatus, 'Models ready...')
