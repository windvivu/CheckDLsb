import os
import pandas as pd
import numpy as np
import functions.myredis as rf
from charts.decorchart import addATR, addMAFast, addMASlow
name = os.path.basename(__file__).split('.py')[0]
# Volatility Indicator
# https://ta-lib.github.io/ta-lib-python/func_groups/volatility_indicators.html

# Đặt thời gian cho MA và ATR
ma_fast_period = 50 
ma_slow_period = 200
atr_period = 14
# atr_threshold = x # Xác định ngưỡng biến động thị trường

# Lưu ý các từ khoá không được có trong tên cột thông thường để tránh nhầm lỗi: sub, sig_, signal_, line_, signallin_
# Tên module tránh khới đầu bằng các từ: ADX, MA, ATR, EMA, SMA
# gợi ý điểm (b/s) mua bán sơ cấp - vẽ cho các cột signal_mod
# đường gợi ý mua bán sơ cấp - vẽ cho các cột signallin_mod
# điểm dấu (x) cho các cột dạng mô hình nến, cột có tên sig_mod
# đường line, cột có tên line_mod:xxlinenamexx , line_mod:xxlinenamexx_sub2
# tên cột(line) có dạng line_lod:xxlinenamexx , hoặc có thêm hậu tố sub để biết đường nằm ở đâu. ở subplot volumn là sub2, plot extra là sub3, không có sub1 hay sub

def findsig(data:pd.DataFrame, testing = False):
    '''
    testing = True để khi muốn lấy ra toàn bộ df để kiểm tra
    '''
    data, colnameMAFast = addMAFast(data, name, ma_fast_period)
    data, colnameMASlow = addMASlow(data, name, ma_slow_period)

    # Tính ATR
    data, colnameATR = addATR(data, name, atr_period)

    # Tính ATR thresold
    atr_values = data[colnameATR].dropna()
    data[f'line_{name}:ATHR_sub3']= np.percentile(atr_values, 75) # lấy ở khoảng 75 percentile

    data['signallin_'+name] = 0.0
    for i in range(0, len(data)):
        # Xác định tín hiệu mua
        if (data[colnameMAFast][i] >= data[colnameMASlow][i]):# & (data[colnameATR][i] <= data[f'line_{name}:ATHR_sub3'][i]):
            data.iloc[i, data.columns.get_loc('signallin_'+name)] = 0.5
        # Xác định tín hiệu bán
        elif (data[colnameMAFast][i] < data[colnameMASlow][i]):# & (data[colnameATR][i] <= data[f'line_{name}:ATHR_sub3'][i]):
            data.iloc[i, data.columns.get_loc('signallin_'+name)] = -0.5

    data = _normalize_df(data)
    dfsigs = data[['timestamp', colnameMAFast, colnameMASlow, colnameATR, f'line_{name}:ATHR_sub3']]
    dfsignal = data[['timestamp', 'signallin_'+name]]

    if testing:
        return dfsigs, dfsignal, data
    else:
        return dfsigs, dfsignal

#---- hàm chuẩn hoá kiểu dữ liệu timestamp--------------------------------------
def _normalize_df(df:pd.DataFrame):
    '''
    Chuẩn hoá kiểu dữ liệu timestamp
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
    return dfsignal.iloc[-1]['signallin_'+module_name]









