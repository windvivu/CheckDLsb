import redis
import json
import pandas as pd
import functions.en_decode as ed

p = ed.mydecodeb64('YlZKcWFuTndZa1ZRVWZodHVZdGdkMnRqVlRaNGJtOXZRbkpsYTFoeVpUWlNRMDVDZGpSdGNtY3dRM0k0UlhaNmJGQlBhVkZMYmtJPQ==', 'fhtuYtgd')
# r = redis.Redis(host='vnn.ink', port=6379, db=0, password=p)
r = redis.Redis(host='127.0.0.1', port=6379, db=0, password=p)

# check connected to redis
def check_connected():
    return r.ping()

# modify key

def add_key_value(key, value):
    r.set(key, value)

def get_value(key):
    return r.get(key)

def get_key(key):
    return r.keys(key)

def delete_key(key):
    r.delete(key)

def expire_key(key, time):
    r.expire(key, time)

# modify df

def add_df(hash_key, df):
    for index, row in df.iterrows():
        r.hset(hash_key, index, row.to_json())

def get_df(hash_key):
    keys = r.hkeys(hash_key)
    data = []
    for key in keys:
        data.append(json.loads(r.hget(hash_key, key)))
    return pd.DataFrame(data)

def delete_df(hash_key):
    r.delete(hash_key)

# modify df_olddt
# hết sức lưu ý về sắp xếp thứ tự khi lấy df ra từ redis
# sau khi sắp xếp cần phải reset lại index của df, nếu không lấy iloc sẽ không như ý muốn

def add_df_olddt(df, symbol):
    for _, row in df.iterrows():
        r.hset('olddt_' + symbol + ':dt', str(row['timestamp']), row.to_json())

def check_exist_olddt(symbol):
    return r.exists('olddt_' + symbol + ':dt')

def get_df_olddt(symbol):
    keys = r.hkeys('olddt_' + symbol + ':dt')
    data = []
    for key in keys:
        data.append(json.loads(r.hget('olddt_' + symbol + ':dt', key)))
    
    df = pd.DataFrame(data)
    if len(data)>0: df = df.sort_values(by='timestamp', ascending=True)
    df.reset_index(drop=True, inplace=True)
    return pd.DataFrame(df)

def get_df_olddt_fromTS(symbol, timestamp):
    keys = r.hkeys('olddt_' + symbol + ':dt')
    keys = [key for key in keys if int(key) > timestamp]
    data = []
    for key in keys:
        data.append(json.loads(r.hget('olddt_' + symbol + ':dt', key)))
    
    df = pd.DataFrame(data)
    if len(data)>0: df = df.sort_values(by='timestamp', ascending=True)
    df.reset_index(drop=True, inplace=True)
    return pd.DataFrame(df)

def get_df_olddt_in_range_sharp(symbol, fromTS, toTS):
    keys = r.hkeys('olddt_' + symbol + ':dt')
    keys = [key for key in keys if int(key) >= fromTS and int(key) <= toTS]
    data = []
    for key in keys:
        data.append(json.loads(r.hget('olddt_' + symbol + ':dt', key)))
    
    df = pd.DataFrame(data)
    if len(data)>0: df = df.sort_values(by='timestamp', ascending=True)
    df.reset_index(drop=True, inplace=True)
    return pd.DataFrame(df)

def remove_hash_olddt(symbol):
    hash_keys = r.keys('olddt_' + symbol + ':*')
    for key in hash_keys:
        r.delete(key)

def remove_log_symbol(symbol):
    hash_keys = r.keys('log_' + symbol + ':*')
    for key in hash_keys:
        r.delete(key)

# modify dfsig

def check_exist_df_sig(symbol, module_name):
    return r.exists('olddt_' + symbol + ':sig:' + module_name)

def add_df_sig(df, symbol, module_name):
    for _, row in df.iterrows():
        r.hset('olddt_' + symbol + ':sig:' + module_name, str(row['timestamp']), row.to_json())

def get_df_sig(symbol, module_name, TS=0):
    keys = r.hkeys('olddt_' + symbol + ':sig:' + module_name)
    keys = [key for key in keys if int(key) > TS]
    data = []
    for key in keys:
        data.append(json.loads(r.hget('olddt_' + symbol + ':sig:' + module_name, key)))
    
    df = pd.DataFrame(data)
    if len(data)>0: df = df.sort_values(by='timestamp', ascending=True)
    df.reset_index(drop=True, inplace=True)
    return df

# modify dfsignal

def add_df_signal(df, symbol, module_name):
   for _, row in df.iterrows():
        r.hset('olddt_' + symbol + ':signal:' + module_name, str(row['timestamp']), row.to_json()) 

def get_df_signal(symbol, module_name, TS=0):
    keys = r.hkeys('olddt_' + symbol + ':signal:' + module_name)
    keys = [key for key in keys if int(key) > TS]
    data = []
    for key in keys:
        data.append(json.loads(r.hget('olddt_' + symbol + ':signal:' + module_name, key)))
    
    df = pd.DataFrame(data)
    if len(data)>0: df = df.sort_values(by='timestamp', ascending=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_df_buy_order(symbol, ts):
    r.hset('olddt_' + symbol + ':order', ts, 'buy')

def add_df_position_up(symbol, ts):
    r.hset('olddt_' + symbol + ':position', ts, '▲pos↑')

def add_df_position_number_up(symbol, ts):
    r.hset('olddt_' + symbol + ':position_num', ts, 1)

def add_df_sell_order(symbol, ts):
    r.hset('olddt_' + symbol + ':order', ts, 'sell')

def add_df_position_down(symbol, ts):
    r.hset('olddt_' + symbol + ':position', ts, '▼pos↓')

def add_df_position_number_down(symbol, ts):
    r.hset('olddt_' + symbol + ':position_num', ts, -1)

def get_df_order(symbol, TS=0):
    keys = r.hkeys('olddt_' + symbol + ':order')
    keys = [key for key in keys if int(key) > TS]
    timestamp = []
    data = []
    for key in keys:
        timestamp.append(int(key))
        data.append(r.hget('olddt_' + symbol + ':order', key).decode('utf-8'))
    df = pd.DataFrame({'timestamp': timestamp, 'order': data})

    if len(data)>0: df = df.sort_values(by='timestamp', ascending=True)
    df.reset_index(drop=True, inplace=True)
    return df

def get_df_position(symbol, TS=0):
    keys = r.hkeys('olddt_' + symbol + ':position')
    keys = [key for key in keys if int(key) > TS]
    timestamp = []
    data = []
    for key in keys:
        timestamp.append(int(key))
        data.append(r.hget('olddt_' + symbol + ':position', key).decode('utf-8'))
    df = pd.DataFrame({'timestamp': timestamp, 'position': data})

    if len(data)>0: df = df.sort_values(by='timestamp', ascending=True)
    df.reset_index(drop=True, inplace=True)
    return df

# modify hash

def add_hash(hash_key, key, value):
    if isinstance(value, bool):
        value = str(value)
    r.hset(hash_key, key, value)

def get_hash(hash_key, key):
    return r.hget(hash_key, key)

def get_hash_string(hash_key, key):
    value = r.hget(hash_key, key)
    if value is not None:
        return value.decode('utf-8')
    else:
        return ''

def get_hash_boolean(hash_key, key):
    value = r.hget(hash_key, key)
    if value is not None:
        value = value.decode('utf-8')
        if value == 'True':
            return True
        elif value == 'False':
            return False
    return value


def delete_hash_key(hash_key, key):
    r.hdel(hash_key, key)

def delete_hash(hash_key):
    r.delete(hash_key)

def check_hash_key(hash_key, key):
    return r.hexists(hash_key, key)

def check_hash(hash_key):
    return r.exists(hash_key)

def expire_hash(hash_key, time):
    r.expire(hash_key, time)

# get dataframe with signal within modulename

def export_df_one_module(symbol, module_name):
    df = get_df_olddt(symbol)
    # kiểm tra nếu có hash 'olddt_symbol:sig:module_name' thì thêm vào df theo patern sig
    if check_hash(f'olddt_{symbol}:sig:{module_name}'):
        dfsig = get_df_sig(symbol, module_name)
        dfsignal = get_df_signal(symbol, module_name)
        df = pd.merge(df, dfsig, on='timestamp', how='left')
        df = pd.merge(df, dfsignal, on='timestamp', how='left')
        if 'sig_'+module_name in df.columns: df['sig_'+module_name] = df['sig_'+module_name].fillna(0)
        if 'signal_'+module_name in df.columns: df['signal_'+module_name] = df['signal_'+module_name].fillna(0)
        if 'line_'+module_name in df.columns: df['line_'+module_name] = df['line_'+module_name].fillna(0)
        if 'signallin_'+module_name in df.columns: df['signallin_'+module_name] = df['signallin_'+module_name].fillna(0)

        # df['signal_'+module_name] = df['signal_'+module_name].astype(int)
    return df

def export_df_all_module(symbol, columnOrder=False, TS=0):
    if TS <= 0:
        df = get_df_olddt(symbol)
    else:
        df = get_df_olddt_fromTS(symbol, TS)
    modules = get_all_modules(symbol)
    # chuyển timestamp sang datetime, cộng thêm giờ GMT tiếng
    GMTs = int(get_hash('info_' + symbol, 'GMTs'))
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms') + pd.Timedelta(hours=GMTs)
    for module in modules:
        # kiểm tra nếu có hash 'olddt_symbol:sig:module_name' thì thêm vào df theo patern sig
        if check_hash(f'olddt_{symbol}:sig:{module}'):
            dfsig = get_df_sig(symbol, module, TS)
            df = pd.merge(df, dfsig, on='timestamp', how='left')
            dfsignal = get_df_signal(symbol, module, TS)
            df = pd.merge(df, dfsignal, on='timestamp', how='left')
            if 'sig_'+module in df.columns: df['sig_'+module] = df['sig_'+module].fillna(0)
            if 'signal_'+module in df.columns: df['signal_'+module] = df['signal_'+module].fillna(0)
            if 'line_'+module in df.columns: df['line_'+module] = df['line_'+module].fillna(0)
            if 'signallin_'+module in df.columns: df['signallin_'+module] = df['signallin_'+module].fillna(0)
            # df['signal_'+module] = df['signal_'+module].astype(int)
    if columnOrder == False:
        return df
    if check_hash(f'olddt_{symbol}:order'):
        dforder = get_df_order(symbol, TS)
        df = pd.merge(df, dforder, on='timestamp', how='left')
        df['order'] = df['order'].fillna('')
    return df

# get all module

def get_all_module_run(symbol):
    '''
    Lấy từ redis tất cả các module đang chạy detect signal
    '''
    keys = r.keys('detect_signal_' + symbol + ':*')
    return [key.decode('utf-8').split(':')[-1] for key in keys]

def get_all_module_hash(symbol):
    '''
    Lấy từ redis tất cả các hash của module đang chạy và đã tạo signal
    '''
    keys = r.keys('olddt_' + symbol + ':signal:*')
    return [key.decode('utf-8') for key in keys]

def get_all_modules(symbol):
    '''
    Lấy từ redis các tên của module đang chạy và đã tạo signal
    '''
    keys = r.keys('olddt_' + symbol + ':signal:*')
    return [key.decode('utf-8').split(':')[-1] for key in keys]

def get_dtlist():
    keys = r.keys('olddt_*:dt')
    return [key.decode('utf-8') for key in keys]

# set command
def setCommand(command:dict, symbol:str):
    r.hset('info_' + symbol,'cmd', json.dumps(command))

# get command
def getCommand(symbol:str):
    return json.loads(r.hget('info_' + symbol,'cmd'))

# add infor of OF
def addOF(symbol:str, of:dict):
    r.hset('info_' + symbol,'OFpriceInfor', json.dumps(of))

def getOF(symbol:str):
    return json.loads(r.hget('info_' + symbol,'OFpriceInfor'))

# write log
def writeLogModdetect(symbol:str, modname:str, log:dict):
    if check_hash('info_' + symbol) == False: 
        return
    timestamp = int(pd.Timestamp.now().timestamp() * 1000)
    log['host_datetime'] = pd.Timestamp(timestamp, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
    r.hset('log_' + symbol + ':moddetect_' + modname, timestamp, json.dumps(log))

def writeLogOG(symbol:str, log:dict):
    timestamp = int(pd.Timestamp.now().timestamp() * 1000)
    log['host_datetime'] = pd.Timestamp(timestamp, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
    r.hset('log_' + symbol + ':OG',timestamp, json.dumps(log))

# def writeLogOF(symbol:str, log:dict):
#     timestamp = int(pd.Timestamp.now().timestamp() * 1000)
#     log['host_datetime'] = pd.Timestamp(timestamp, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
#     r.hset('log_' + symbol + ':OF',timestamp, json.dumps(log))