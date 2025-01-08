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
        sig = ta.CDLLONGLEGGEDDOJI(dfsub['open'], dfsub['high'], dfsub['low'], dfsub['close'])

        dfsub['sig_'+self.name] = sig
        dfsub['signal_'+self.name] = 0

        isDjlongleg = False
        previous_row = None
        for i in range(0, len(dfsub)):
            if i>0: previous_row = dfsub.iloc[i-1] # lấy previous row
            
            if sig[i] > 0: 
                # sig cho thấy đây là doji longleg thì ghi nhớ tín hiệu, nhảy qua vòng lặp khác, không làm gì thêm
                isDjlongleg = True
                continue 
            else:
                # nếu không phải doji longleg, xem lại tín hiệu trước đó phải doji longleg không, nếu phải thì mới thực hiện tiếp
                if isDjlongleg :
                    # ví tín hiệu nhớ trước đó có, nên cần xoá nhớ tín hiệu trước đó
                    isDjlongleg = False
                    if previous_row is None: continue  # sẽ không làm gì nếu chưa có dữ liệu previous row

                    current_row = dfsub.iloc[i]
            
                    if self._is_bullish_candle(current_row, previous_row):
                        dfsub.iloc[i, dfsub.columns.get_loc('signal_'+self.name)] = 1
                    if self._is_bearish_candle(current_row, previous_row):
                        dfsub.iloc[i, dfsub.columns.get_loc('signal_'+self.name)] = -1

        dfsub = self._normalize_df(dfsub)
        self.dfsigs = dfsub[['timestamp', 'sig_'+self.name]]
        self.dfsignal = dfsub[['timestamp', 'signal_'+self.name]]

        if self.testing:
            return self.dfsigs, self.dfsignal, dfsub
        else:
            return self.dfsigs, self.dfsignal


    #---- các hàm phụ trợ ----------------------------------------------------------

    # define a function to detect up candle
    def _is_bullish_candle(self, current_row, previous_row):
        is_up_candle = False
        if current_row['close'] > current_row['open']: is_up_candle = True
        is_current_close_higher_than_previous_close = False
        if current_row['close'] > previous_row['close']: is_current_close_higher_than_previous_close = True
        # is_previous_row_down = False
        # if previous_row['close'] <= previous_row['open']: is_previous_row_down = True
        
        return is_up_candle and is_current_close_higher_than_previous_close #and is_previous_row_down

    # define a function to detect down candle
    def _is_bearish_candle(self, current_row, previous_row):
        is_down_candle = False
        if current_row['close'] < current_row['open']: is_down_candle = True
        is_current_close_lower_than_previous_close = False
        if current_row['close'] < previous_row['close']: is_current_close_lower_than_previous_close = True
        # is_previous_row_up = False
        # if previous_row['close'] >= previous_row['open']: is_previous_row_up = True
        
        return is_down_candle and is_current_close_lower_than_previous_close # and is_previous_row_up


