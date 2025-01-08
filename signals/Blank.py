# import talib as ta
import os
import pandas as pd
import time
# import sys
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)
from signals._sigCore import sigcore
import functions.myredis as rf

# https://ta-lib.github.io/ta-lib-python/

class SigExt(sigcore):
    def __init__(self, symbol, tf, testing:bool=False):
        '''
        https://ta-lib.github.io/ta-lib-python/
        
        '''
        super().__init__(symbol, tf, testing)
        self.name = os.path.basename(__file__).split('.py')[0]


    def findsig(self, dfsub:pd.DataFrame, para:dict={}, hashkeystatus:str=''):
        
        #---- thuật toán tìm sig/line_ và signal_/signallin_ ở đây

        dfsub['sig_'+self.name] = 0
        dfsub['signal_'+self.name] = 0

        #---------------------------------------------------------
        dfsub = self._normalize_df(dfsub)
        self.dfsigs = dfsub[['timestamp', 'sig_'+self.name]]
        self.dfsignal = dfsub[['timestamp', 'signal_'+self.name]]

        if self.testing:
            return self.dfsigs, self.dfsignal, dfsub
        else:
            return self.dfsigs, self.dfsignal
        
    def firstInit(self, hashkeystatus:str=''):
        for i in range(2):
            if hashkeystatus != '': rf.add_key_value(hashkeystatus, f'firstInit {i}')  
            time.sleep(1)
    
    def secondInit(self, hashkeystatus:str=''):
        for i in range(2):
            if hashkeystatus != '': rf.add_key_value(hashkeystatus, f'secondInit {i}')  
            time.sleep(1)