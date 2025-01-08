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
        Chỉ để test
        '''
        super().__init__(symbol, tf, market, testing)
        self.name = os.path.basename(__file__).split('.py')[0]


    def findsig(self, dfsub:pd.DataFrame, para:dict={}, hashkeystatus:str=''):
        self.calculate_rsi(dfsub)
        self.calculate_momentum(dfsub, period=14)
        dfsub['signal_'+self.name] = 0

        dfsub = self._normalize_df(dfsub)
        self.dfsigs = dfsub[['timestamp', 'line_'+self.name+':RSI_sub3']]
        self.dfsignal = dfsub[['timestamp', 'signal_'+self.name]]

        if self.testing:
            return self.dfsigs, self.dfsignal, dfsub
        else:
            return self.dfsigs, self.dfsignal


    #---- các hàm phụ trợ ----------------------------------------------------------

    def calculate_rsi(self, data, period=14):
        delta = data['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        RS = gain / loss
        data['line_'+self.name+':RSI_sub3'] = 100 - (100 / (1 + RS))
        data['line_'+self.name+':RSI_sub3'] = data['line_'+self.name+':RSI_sub3'].bfill().ffill()
        data['line_'+self.name+':RSIref70_sub3'] = 70
        data['line_'+self.name+':RSIref30_sub3'] = 30
    
    def calculate_momentum(self, data, period=14):
        data['line_'+self.name+':Momentum_sub3'] = data['close'].diff(period)
        # Hoặc Momentum = Giá thị trường hiện tại – Giá trị trung bình trong một khoảng thời gian nhất định
        data['line_'+self.name+':Momentum_sub3'] = data['line_'+self.name+':Momentum_sub3'].bfill().ffill()
        data['line_'+self.name+':Mo_sub3'] = 0
    



