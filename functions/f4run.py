import os
import functions.myredis as rf
import pandas as pd
import json
import requests
from datetime import datetime

def _clearScreen():
    os.system('cls' if os.name == 'nt' else 'clear')

def __slogfilexlsx(df: pd.DataFrame, filename):
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    saved = False
    if df.empty == False:
        df.to_excel(writer, index=False, sheet_name='General')
        saved = True
    writer.close()
    return saved

def _calSignal(symbol, ogts): # tính toán điểm mua bán (ogts: order goods timestamp)
    hm = rf.get_all_module_hash(symbol)
    # lấy ra bảng chứa tên và điểm tương ứng mỗi module
    tableModules = {}
    for m in hm: # duyệt qua mỗi module hash
        # bị lỗi chỗ này, không tìm thấy haskey. thực tế trong redis có dạng xxxx.0 , trong khi đang tìm xxxx
        # đã gặp lỗi này khi pandas tự đổi kiểu dữ liệu float cho cột timestamp
        if rf.check_hash_key(m,str(ogts)):        
            j = rf.get_hash(m,str(ogts))
            data_dict = json.loads(j.decode('utf-8'))
            modname = m.split(':')[-1]
            if f'signal_{modname}' in data_dict:
                point =  float(data_dict[f'signal_{modname}'])
            if f'signallin_{modname}' in data_dict:
                point =  float(data_dict[f'signallin_{modname}'])
            tableModules[modname] = point
    
    # tính điểm
    sum = 0
    for val in tableModules.values():
        sum += val
    return sum

def cSymbol(symbol:str) -> str:
    return symbol.split(':')[0]


def sendTelegramMessage(message):
    TOKEN = "7237333197:AAG_DXeJk6rE_hxFcgJpPmnuEfZu0ejGkYw"
    chat_id = "5064020009"
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    message = f"[{dt_string}]:\n{message}"
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
        requests.get(url)
    except Exception as e:
        print('\tError:', e)

def writeError(err, at_ts:int, filename:str):
    import time
    if not os.path.exists('./error'):
        os.makedirs('./error')
    with open(f'./error/error_{filename}.txt', 'a', encoding='utf-8') as f:
        f.write(f'{time.strftime("-- %Y-%m-%d %H:%M:%S --", time.localtime())}\nAt TS: {at_ts}\n{err}\n\n')
