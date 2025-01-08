# File này dùng làm thư viện cho các file khác để kết nối và truy vấn dữ liệu từ local database sqlite
# local database này được tạo từ dữ liệu của sàn Binance, làm cache để không phải tải lại dữ liệu từ sàn

import sqlite3
import os
import pandas as pd
import datetime
import random

def _runSQLandfetch(conn, sql, fetch = 0):
    '''
    fetch <= 0 : all
    fetch = 1 : one
    fetch > 1 : fetchmany(fetch)
    '''
    cursor = conn.cursor()
    cursor.execute(sql)
    if fetch <= 0:
        return cursor.fetchall()
    elif fetch == 1:
        return cursor.fetchone()
    else:
        return cursor.fetchmany(fetch)

def saveDataToDB(pathData, symbol, timeframe, newdt, silent = False):
    '''
    pathData: path to folder containing database files, not database file
    symbol: pair of cryptocurrency, e.g. 'BTC/USDT'
    newdt: dataframe to save to database
    '''
    # connect to database
    filedb = symbol.replace('/','_') + '.db'
    if not os.path.exists(os.path.join(pathData, filedb)):
        print('Error: Database not found!')   
        return None
    else:
        conn = sqlite3.connect(os.path.join(pathData, filedb))
        newdt.to_sql(f'tf{timeframe}', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()
        if silent == False: print('Data saved to database.')

def getDTsqlite(pathData, symbol, timeframe, since_ts= None, end_ts = None, silent = False):
    '''
    pathData: path to folder containing database files, not database file
    symbol: pair of cryptocurrency, e.g. 'BTC/USDT'
    since_ts: timestamp in ms seconds, e.g. 1617244800000
    '''
    # connect to database
    filedb = symbol.replace('/','_') + '.db'
    if not os.path.exists(os.path.join(pathData, filedb)):
        print('Error: Database not found!')   
        return None
    else:
        if silent == False: print('Connecting to database...')
        conn = sqlite3.connect(os.path.join(pathData, filedb))
        if silent == False: print('Connected to database. Fetching data...')
        if since_ts is not None and end_ts is None:
            where = f"WHERE timestamp >= {int(since_ts)}"
        elif since_ts is not None and end_ts is not None:
            where = f"WHERE timestamp >= {int(since_ts)} AND timestamp <= {int(end_ts)}"
        elif since_ts is None and end_ts is not None:
            where = f"WHERE timestamp <= {int(end_ts)}"
        else:
            where = ""   
        dt = _runSQLandfetch(conn, f"SELECT * FROM tf{timeframe}" + " " + where + ";", -1)
        df = pd.DataFrame(dt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume','datetime'])
        conn.close()
        return df

def getDBtables(pathData, symbol):
    filedb = symbol.replace('/','_') + '.db'
    if not os.path.exists(os.path.join(pathData, filedb)):
        print('Error: Database not found!')   
        return None
    else:
        print('Connecting to database...')
        conn = sqlite3.connect(os.path.join(pathData, filedb))
        print('Connected to database. Fetching data...')
        dt = _runSQLandfetch(conn, "SELECT name FROM sqlite_master WHERE type='table';", -1)
        conn.close()
        return dt

def getMaxMinDateDB(pathData, symbol, timeframe, convertTo = 'stringdate'):
    '''return max and min timestamp from database
        default: convertTo = 'stringdate' -> strftime('%Y-%m-%d %H:%M:%S')
        convertTo = 'datetime' -> datetime.datetime
        convertTo = 'timestamp' -> timestamp
    '''
    filedb = symbol.replace('/','_') + '.db'
    if not os.path.exists(os.path.join(pathData, filedb)):
        print('Error: Database not found!')   
        return None, None
    else:
        # print('Connecting to database...')
        conn = sqlite3.connect(os.path.join(pathData, filedb))
        # print('Connected to database. Fetching data...')
        dt = _runSQLandfetch(conn, f"SELECT MAX(timestamp), MIN(timestamp) FROM tf{timeframe};", 1)
        conn.close()
        maxts, mints = dt
        if convertTo == 'timestamp':
            return maxts, mints
        if convertTo == 'datetime':
            maxts = datetime.datetime.fromtimestamp(maxts/1000)
            mints = datetime.datetime.fromtimestamp(mints/1000)
            return min, max
        if convertTo == 'stringdate':
            maxts = datetime.datetime.fromtimestamp(maxts/1000)
            maxts = maxts.strftime('%Y-%m-%d %H:%M:%S')
            mints = datetime.datetime.fromtimestamp(mints/1000)
            mints = mints.strftime('%Y-%m-%d %H:%M:%S')
        return maxts, mints

def getMaxTimestamp(pathData, symbol, timeframe):
    '''return max timestamp from database'''
    filedb = symbol.replace('/','_') + '.db'
    if not os.path.exists(os.path.join(pathData, filedb)):
        print('Error: Database not found!')   
        return None
    else:
        # print('Connecting to database...')
        conn = sqlite3.connect(os.path.join(pathData, filedb))
        # print('Connected to database. Fetching data...')
        dt = _runSQLandfetch(conn, f"SELECT MAX(timestamp) FROM tf{timeframe};", 1)
        conn.close()
        max = dt[0]
        return max

def removeRowDB(pathData, symbol, timeframe, timestamp):
    '''Remove row from database'''
    filedb = symbol.replace('/','_') + '.db'
    if not os.path.exists(os.path.join(pathData, filedb)):
        print('Error: Database not found!')   
        return None
    else:
        # print('Connecting to database...')
        conn = sqlite3.connect(os.path.join(pathData, filedb))
        # print('Connected to database. Fetching data...')
        conn.execute(f"DELETE FROM tf{timeframe} WHERE timestamp = {timestamp};")
        conn.commit()
        conn.close()
    
def getGapCandle(pathData, symbol, timeframe):
    '''Xem 2 nến cách nhau bao nhiêu time stamp'''
    filedb = symbol.replace('/','_') + '.db'
    if not os.path.exists(os.path.join(pathData, filedb)):
        print('Error: Database not found!')   
        return None
    else:
        # print('Connecting to database...')
        conn = sqlite3.connect(os.path.join(pathData, filedb))
        # print('Connected to database. Fetching data...')
        dt = _runSQLandfetch(conn, f"SELECT timestamp FROM tf{timeframe} LIMIT 2;", -1)
        conn.close()
        return dt[1][0] - dt[0][0]

def getRandomPeriod(pathData, symbol, timeframe, num_of_day, to_string = False):
    '''
    Lấy khoảng thời gian ngẫu nhiên trong database, có đầu vào là số ngày cho trước
    Trả về 2 giá trị timestamp hoặc string date
    '''
    maxTs, minTs = getMaxMinDateDB(pathData, symbol, timeframe, convertTo = 'timestamp')
    if maxTs is None or minTs is None:
        return None, None
    # get random number between max and min
    startTs = random.randrange(minTs, maxTs, step=60*1000)
    endTs = startTs + datetime.timedelta(days=num_of_day).total_seconds()*1000
    if endTs > maxTs:
        endTs = maxTs
        startTs = endTs - datetime.timedelta(days=num_of_day).total_seconds()*1000
    # convert timestamp to string date
    if to_string == True:
        startTs = datetime.datetime.fromtimestamp(startTs/1000).strftime('%d/%m/%Y %H:00')
        endTs = datetime.datetime.fromtimestamp(endTs/1000).strftime('%d/%m/%Y %H:00')
    return startTs , endTs

def getRandomData(pathData, symbol, timeframe, num_of_day):
    '''
    Trả về data trong khoảng thời gian ngẫu nhiên, với số ngày cho trước
    '''
    filedb = symbol.replace('/','_') + '.db'
    if not os.path.exists(os.path.join(pathData, filedb)):
        print('Error: Database not found!')   
        return None
    else:
        # print('Connecting to database...')
        conn = sqlite3.connect(os.path.join(pathData, filedb))
        # print('Connected to database. Fetching data...')
        maxTs, minTs = _runSQLandfetch(conn, f"SELECT MAX(timestamp), MIN(timestamp) FROM tf{timeframe};", 1)
        # get random number between max and min
        startTs = random.randrange(minTs, maxTs, step=60*1000)
        endTs = startTs + datetime.timedelta(days=num_of_day).total_seconds()*1000
        if endTs > maxTs:
            endTs = maxTs
            startTs = endTs - datetime.timedelta(days=num_of_day).total_seconds()*1000
        data = None
        try:
            data = pd.read_sql(f"SELECT * FROM tf{timeframe} WHERE timestamp >= {startTs} AND timestamp <= {endTs}", conn)
        except Exception as ex:
            print(ex)
        finally:
            conn.close()
        return data

def getRandomDataCandle(pathData, symbol, timeframe, num_of_candle):
    '''
    Trả về data ngẫu nhiên với số nến cho trước, cố định số nến
    '''
    filedb = symbol.replace('/','_') + '.db'
    if not os.path.exists(os.path.join(pathData, filedb)):
        print('Error: Database not found!')   
        return None
    else:
        gapCandle = getGapCandle(pathData, symbol, timeframe)
        # print('Connecting to database...')
        conn = sqlite3.connect(os.path.join(pathData, filedb))
        # print('Connected to database. Fetching data...')
        maxTs, minTs = _runSQLandfetch(conn, f"SELECT MAX(timestamp), MIN(timestamp) FROM tf{timeframe};", 1)
        # get random number between max and min
        startTs = random.randrange(minTs, maxTs, step=60*1000)
        endTs = startTs + num_of_candle*gapCandle
        if endTs > maxTs:
            endTs = maxTs
            startTs = endTs - num_of_candle*gapCandle
        data = None
        try:
            data = pd.read_sql(f"SELECT * FROM tf{timeframe} WHERE timestamp >= {startTs} AND timestamp <= {endTs}", conn)
        except Exception as ex:
            print(ex)
        finally:
            conn.close()
        return data