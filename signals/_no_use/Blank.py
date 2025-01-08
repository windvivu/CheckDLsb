# import talib as ta
import os
import pandas as pd
import functions.myredis as rf
name = os.path.basename(__file__).split('.py')[0]

# https://ta-lib.github.io/ta-lib-python/
# https://ta-lib.github.io/ta-lib-python/func_groups/pattern_recognition.html

# Lưu ý các từ khoá không được có trong tên cột thông thường để tránh nhầm lỗi: sub, sig_, signal_, line_, signallin_
# gợi ý điểm (b/s) mua bán sơ cấp - vẽ cho các cột signal_mod
# đường gợi ý mua bán sơ cấp - vẽ cho các cột signallin_mod
# điểm dấu (x) cho các cột dạng mô hình nến, cột có tên sig_mod
# đường line, cột có tên line_mod:xxlinenamexx , line_mod:xxlinenamexx_sub2
# tên cột(line) có dạng line_lod:xxlinenamexx , hoặc có thêm hậu tố sub để biết đường nằm ở đâu. ở subplot volumn là sub2, plot extra là sub3, không có sub1 hay sub

def findsig(df:pd.DataFrame, testing = False):
    '''
    testing = True để khi muốn lấy ra toàn bộ df để kiểm tra
    ''' 

    df['sig_'+name] = 0
    df['signal_'+name] = 0

    df = _normalize_df(df)
    dfsigs = df[['timestamp', 'sig_'+name]]
    dfsignal = df[['timestamp', 'signal_'+name]]
    if testing:
        return dfsigs, dfsignal, df
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
    return dfsignal.iloc[-1]['signal_'+module_name]

#---- các hàm phụ trợ ----------------------------------------------------------



