import pandas as pd
import numpy as np
import functions.myredis as rf
import os
from charts.decorchart import addMA, addMA, addPeakTrough

name = os.path.basename(__file__).split('.py')[0]
# Dựa vào đường cơ bản nhất để tính toán

# Lưu ý các từ khoá không được có trong tên cột thông thường để tránh nhầm lỗi: sub, sig_, signal_, line_, signallin_
# Tên module tránh khới đầu bằng các từ: ADX, MA, ATR, EMA, SMA
# gợi ý điểm (b/s) mua bán sơ cấp - vẽ cho các cột signal_mod
# đường gợi ý mua bán sơ cấp - vẽ cho các cột signallin_mod
# điểm dấu (x) cho các cột dạng mô hình nến, cột có tên sig_mod
# đường line, cột có tên line_mod:xxlinenamexx , line_mod:xxlinenamexx_sub2
# tên cột(line) có dạng line_lod:xxlinenamexx , hoặc có thêm hậu tố sub để biết đường nằm ở đâu. ở subplot volumn là sub2, plot extra là sub3, không có sub1 hay sub

def findsig(df:pd.DataFrame, testing = False, para={'span':5, 'smooth':5}):
    '''
    testing = True để khi muốn lấy ra toàn bộ df để kiểm tra
    '''
    _modulename = name 
    df, colnameMA = addMA(df, modulename=_modulename, para=para['span'], smooth=para['smooth'])
    
    df, colPeak, colTrough = addPeakTrough(df, colnameMA, returnColPeakTrough=True)

    df['sig_'+name] = np.where((df[colPeak] != 0) | (df[colTrough] != 0), df[colPeak] + df[colTrough], 0)

    # Xoá 2 cột Peak và Trough vì không cần nữa
    df.drop([colPeak, colTrough], axis=1, inplace=True)

    conditions = [
    (df['sig_'+name] > 0),
    (df['sig_'+name] < 0),
    (df['sig_'+name] == 0)
    ]

    choices = [-1, 1, 0]

    df['signal_'+name] = np.select(conditions, choices, default=0)

    df = _normalize_df(df)
    dfsigs = df[['timestamp', 'sig_'+name, colnameMA]]
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