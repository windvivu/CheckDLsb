import talib as ta
import os
import pandas as pd
import functions.myredis as rf
from charts.decorchart import addAverageVolume, addMAFast, addMASlow

name = os.path.basename(__file__).split('.py')[0]

# https://ta-lib.github.io/ta-lib-python/
# https://ta-lib.github.io/ta-lib-python/func_groups/pattern_recognition.html

# Lưu ý các từ khoá không được có trong tên cột thông thường để tránh nhầm lỗi: sub, sig_, signal_, line_, signallin_
# Tên module tránh khới đầu bằng các từ: ADX, MA, ATR, EMA, SMA
# gợi ý điểm (b/s) mua bán sơ cấp - vẽ cho các cột signal_mod
# đường gợi ý mua bán sơ cấp - vẽ cho các cột signallin_mod
# điểm dấu (x) cho các cột dạng mô hình nến, cột có tên sig_mod
# đường line, cột có tên line_mod:xxlinenamexx , line_mod:xxlinenamexx_sub2
# tên cột(line) có dạng line_lod:xxlinenamexx , hoặc có thêm hậu tố sub để biết đường nằm ở đâu. ở subplot volumn là sub2, plot extra là sub3, không có sub1 hay sub

def findsig(df:pd.DataFrame, testing = False):
    '''
    testing = True để khi muốn lấy ra toàn bộ df để kiểm tra
    ''' 

    sig = ta.CDLLONGLEGGEDDOJI(df['open'], df['high'], df['low'], df['close'])

    df['sig_'+name] = sig

    df, mafastcol = addMAFast(df, name)
    df, maslowcol = addMASlow(df, name)    
    df, av_colname = addAverageVolume(df, name, 10)
    
    df['signal_'+name] = 0.0


    isDjlongleg = False
    previous_row = None
    for i in range(0, len(df)):
        if i>0: previous_row = df.iloc[i-1] # lấy previous row
        
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

                current_row = df.iloc[i]
                
                if _is_bullish_candle(current_row, previous_row) and _is_volume_higher_than_avolume(current_row, av_colname):
                    df.iloc[i, df.columns.get_loc('signal_'+name)] = 1
                if _is_bearish_candle(current_row, previous_row) and _is_volume_higher_than_avolume(current_row, av_colname):
                    df.iloc[i, df.columns.get_loc('signal_'+name)] = -1

    
    df = _normalize_df(df)
    
    dfsigs = df[['timestamp', 'sig_'+name, av_colname, mafastcol, maslowcol]]
    dfsignal = df[['timestamp', 'signal_'+name]]
    if testing:
        return dfsigs, dfsignal, df
    else:
        return dfsigs, dfsignal
    
#---- hàm chuẩn hoá kiểu dữ liệu timestamp--------------------------------------
def _normalize_df(df:pd.DataFrame):
    '''
    Chuẩn hoá kiểu dữ liệu timestamp
    Giữ nguyên kiểu dữ liệu của cột timestamp, nếu không nó sẽ bị pandas đổi kiểu sau khi thêm cột float, gây lỗi không mong muốn
    '''
    df['timestamp'] = df['timestamp'].astype('Int64')
    return df

#---- hàm ghi vào redis---------------------------------------------------------
def save_signal_to_redis(df:pd.DataFrame, symbol:str, module_name:str):
    '''
    Ghi tín hiệu vào redis
    '''
    limit = 5
    dfsigs, dfsignal = findsig(df)
    # nếu đã tồn tại hash sig thì chỉ ghi giới hạn số dòng mới
    if rf.check_exist_df_sig(symbol, module_name):
        rf.add_df_sig(dfsigs.tail(limit), symbol, module_name)
        rf.add_df_signal(dfsignal.tail(limit), symbol, module_name)
    # chưa tồn tại thì ghi toàn bộ
    else:
        rf.add_df_sig(dfsigs, symbol, module_name)
        rf.add_df_signal(dfsignal, symbol, module_name)

    # get value in column signal of last row
    return dfsignal.iloc[-1]['signal_'+module_name]

#---- các hàm phụ trợ ----------------------------------------------------------

# define a function to detect up candle
def _is_bullish_candle(current_row, previous_row):
    is_up_candle = False
    if current_row['close'] > current_row['open']: is_up_candle = True
    is_current_close_higher_than_previous_close = False
    if current_row['close'] > previous_row['close']: is_current_close_higher_than_previous_close = True
    # is_previous_row_down = False
    # if previous_row['close'] <= previous_row['open']: is_previous_row_down = True
    
    return is_up_candle and is_current_close_higher_than_previous_close #and is_previous_row_down

# define a function to detect down candle
def _is_bearish_candle(current_row, previous_row):
    is_down_candle = False
    if current_row['close'] < current_row['open']: is_down_candle = True
    is_current_close_lower_than_previous_close = False
    if current_row['close'] < previous_row['close']: is_current_close_lower_than_previous_close = True
    # is_previous_row_up = False
    # if previous_row['close'] >= previous_row['open']: is_previous_row_up = True
    
    return is_down_candle and is_current_close_lower_than_previous_close # and is_previous_row_up

# xác định volume hiện tại cao hơn volumn trung bình av10
def _is_volume_higher_than_avolume(current_row, _avcolname:str):
    return current_row['volume'] > current_row[_avcolname]


