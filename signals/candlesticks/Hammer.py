import talib as ta
import os
import sys
import pandas as pd
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from signals._sigCore import sigcore

# https://ta-lib.github.io/ta-lib-python/
# https://ta-lib.github.io/ta-lib-python/func_groups/pattern_recognition.html

# Lưu ý các từ khoá không được có trong tên cột thông thường để tránh nhầm lỗi: sub, sig_, signal_, line_, signallin_
# Tên module tránh khới đầu bằng các từ: ADX, MA, ATR, EMA, SMA
# gợi ý điểm (b/s) mua bán sơ cấp - vẽ cho các cột signal_mod
# đường gợi ý mua bán sơ cấp - vẽ cho các cột signallin_mod
# điểm dấu (x) cho các cột dạng mô hình nến, cột có tên sig_mod
# đường line, cột có tên line_mod:xxlinenamexx , line_mod:xxlinenamexx_sub2
# tên cột(line) có dạng line_lod:xxlinenamexx , hoặc có thêm hậu tố sub để biết đường nằm ở đâu. ở subplot volumn là sub2, plot extra là sub3, không có sub1 hay sub

class SigExt(sigcore):
    def __init__(self, symbol, tf, market, testing:bool=False):
        '''
        https://ta-lib.github.io/ta-lib-python/
        
        https://ta-lib.github.io/ta-lib-python/func_groups/pattern_recognition.html
        '''
        super().__init__(symbol, tf, market, testing)
        self.name = os.path.basename(__file__).split('.py')[0]


    def findsig(self, dfsub:pd.DataFrame, para:dict={}, hashkeystatus:str=''):
        sig = ta.CDLHAMMER(dfsub['open'], dfsub['high'], dfsub['low'], dfsub['close'])

        dfsub['sig_'+self.name] = sig
        dfsub['signal_'+self.name] = 0

        dfsub = self._normalize_df(dfsub)
        self.dfsigs = dfsub[['timestamp', 'sig_'+self.name]]
        self.dfsignal = dfsub[['timestamp', 'signal_'+self.name]]

        if self.testing:
            return self.dfsigs, self.dfsignal, dfsub
        else:
            return self.dfsigs, self.dfsignal


    #---- các hàm phụ trợ ----------------------------------------------------------

    


