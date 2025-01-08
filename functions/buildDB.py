
import os
import sqlite3
import ccxt
from datetime import datetime
import time
from functions.cvstring2timestamp import cvstring2timestamp
from loadData.downloadData import _coreLoadBinanceData, _coreLoadYfinanceData, _convertDataframe

totalrow = 0
turnon = True

def _signal_handler(var1, var2):
    global turnon
    print('')
    confirm = input("Do you want to stop? (y/n): ")
    if confirm.lower() == 'y':
        turnon = False
    else:
        print('Continue building...')

def _checkTableExist(conn, timeframe):
    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='tf{timeframe}';")
    if cursor.fetchone() is None:
        return False
    return True

# run sql and fetch all
def _runSQLandfetch(conn, sql, fetch = 0):
    cursor = conn.cursor()
    cursor.execute(sql)
    if fetch <= 0:
        return cursor.fetchall()
    elif fetch == 1:
        return cursor.fetchone()
    else:
        return cursor.fetchmany(fetch)
    
def _printSuccess(endx, row_added):
    end_str = datetime.fromtimestamp(endx/1000).strftime('%d/%m/%Y %H:%M')
    print(row_added, 'added to database ( Reached to', end_str, ')')
    
def _removeRowDf(df, timestamp):
    df = df.drop(df[df['timestamp'] == timestamp].index)
    return df

def _loadandsaveData(symbol, conn, timeframe, since, end, drop = None, source='Binance'):
    global totalrow
        
    # core load data
    try:
        if source=='Yfinance':
            data = _coreLoadYfinanceData(symbol, timeframe, since)
        if source=='FBinance':
            data = _coreLoadBinanceData(symbol, timeframe, since, option='future')
        else:
            data = _coreLoadBinanceData(symbol, timeframe, since, option='spot')
    except Exception as e:
        if source=='Yfinance':
            print('Error in loading data from Yfinance:', e)
        else:
            print('Error in loading data from Binance:', e)
        return
    
    if len(data) == 0:
        print('No data found!')
        return
    
    lastTimestamp = data[-1][0]
    if lastTimestamp >= end:
        # slice data to end_timestamp
        while lastTimestamp > end:
            data.pop()
            lastTimestamp = data[-1][0]
        df = _convertDataframe(data)
        if drop != None: df = _removeRowDf(df, drop)
        row_added = df.to_sql('tf'+timeframe, conn, if_exists='append', index=False)
        totalrow += row_added
        _printSuccess(end, row_added)
        return
    
    lastTimestamp +=1
    while lastTimestamp < end:
        if turnon == False:
            print('Stop loading data!!!')
            break
        
        if len(data) > 0:
            df = _convertDataframe(data)
            if drop != None: df = _removeRowDf(df, drop)
            row_added = df.to_sql('tf'+timeframe, conn, if_exists='append', index=False)
            totalrow += row_added
            _printSuccess(lastTimestamp, row_added)

        # core load more data
        try:
            if source=='Yfinance':
                data = _coreLoadYfinanceData(symbol, timeframe, lastTimestamp)
            elif source=='FBinance':
                data = _coreLoadBinanceData(symbol, timeframe, lastTimestamp, option='future')
            else:
                data = _coreLoadBinanceData(symbol, timeframe, lastTimestamp, option='spot')
        except Exception as e:
            if source=='Yfinance':
                print('Error in loading data from Yfinance:', e)
            else:
                print('Error in loading data from Binance:', e)
            data = []
            time.sleep(1)
            continue

        if len(data) == 0:
            break
        else:
            lastTimestamp = data[-1][0] + 1
            df = _convertDataframe(data)
            if drop != None: df = _removeRowDf(df, drop)
            row_added = df.to_sql('tf'+timeframe, conn, if_exists='append', index=False)
            totalrow += row_added
            data = []  # reset về rỗng để khi lặp lại nó không lưu lại data lần nữa
            _printSuccess(lastTimestamp, row_added)

        #time.sleep(0.5)

def BuilDB(symbol = 'BTC/USDT', since = '12/1/2000', end = '12/8/2025', timeframe = '1m', mess='', pathData = '', source='Binance'):
    '''
    Tải dữ liệu từ binance xuống và lưu vào dữ liệu sqlite. Nếu khoảng thời khai báo đã có dữ liệu
    thì không tải lại. Chỉ tải dữ liệu mới nhất so với dữ liệu đang có
    '''    
    # connect to database, if not exist, create new one
    if not os.path.exists(pathData):
        os.makedirs(pathData)
    filedb = symbol.replace('/','_') + '.db'

    if not os.path.exists(os.path.join(pathData, filedb)):
        print(mess,'Creating new database...')
    else:
        print(mess,'Connecting to database...')
    conn = sqlite3.connect(os.path.join(pathData, filedb))

    print(mess,'Begin building data from ', since, 'to', end, '...')

    # standardize timestamp
    since = cvstring2timestamp(since) # đã convert qua timestamp
    end = cvstring2timestamp(end) # đã convert qua timestamp

    if since == 'error' or end == 'error':
        print(mess,'Error in date and time input!')
        conn.close()
        return

    bn = ccxt.binance()
    now_ts = bn.milliseconds()

    if end > now_ts: end = now_ts
    if since > now_ts: since = now_ts
    if end < since:
        print(mess,'End date must be greater than since date')
        conn.close()
        return

    # make file 'filedb.txt' with content doing to mark the filedb is building
    with open(os.path.join(pathData, filedb+'.txt'), 'w') as f:
        f.write('Building data...')

    if _checkTableExist(conn, timeframe):
        min, max = _runSQLandfetch(conn, f"SELECT MIN(timestamp), MAX(timestamp) FROM tf{timeframe};", 1)
        if  min <= since <= end <=max:
            print(mess,'Data already exist in database.')
            conn.close()
            return
        elif since <= min <= end <= max:
            end = min
            _loadandsaveData(symbol, conn, timeframe, since, end, drop=min, source=source)
        elif min <= since <= max <= end:
            since = max
            _loadandsaveData(symbol, conn, timeframe, since, end, drop=max, source=source)
        elif since < min <= max < end:
            original_end = end
            end = min
            _loadandsaveData(symbol, conn, timeframe, since, end, drop=min, source=source)
            end = original_end
            since = max
            _loadandsaveData(symbol, conn, timeframe, since, end, drop=max, source=source)
    else:
        _loadandsaveData(symbol, conn, timeframe, since, end, source=source)

    conn.close()
    print(mess,"Download and save data completed.")
    print(mess,f'Total: {totalrow} rows added to database')

    # remove file 'filedb.txt'
    os.remove(os.path.join(pathData, filedb+'.txt'))