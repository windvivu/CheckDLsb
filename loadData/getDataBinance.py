# File này dùng làm thư viện cho các file khác để lấy dữ liệu ohlcv mới nhất từ sàn Binance

import ccxt
import pandas as pd

def getNewestData(symbol, timeframe, option, limit=1):
    '''
    option: spot or future
    '''
    if option == 'spot':
        exchange = ccxt.binance({
            'enableRateLimit': True
        } )
    else: # future
        exchange = ccxt.binanceusdm({
            'enableRateLimit': True
        } )
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    ohlcv = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    ohlcv['datetime'] = pd.to_datetime(ohlcv['timestamp'], unit='ms')
    return ohlcv

def getNewestDataSince(symbol, timeframe, since_ts, option):
    '''
    option: spot or future
    '''
    if option == 'spot':
        exchange = ccxt.binance({
            'enableRateLimit': True
        } )
    else: # future
        exchange = ccxt.binanceusdm({
            'enableRateLimit': True
        } )
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts)
    except:
        return None
    ohlcv = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    ohlcv['datetime'] = pd.to_datetime(ohlcv['timestamp'], unit='ms')
    return ohlcv

def getNewestDataSLimit(symbol, timeframe, since_ts, showlimit, option):
    '''
    option: spot or future
    '''
    if option == 'spot':
        exchange = ccxt.binance({
            'enableRateLimit': True
        } )
    else: # future
        exchange = ccxt.binanceusdm({
            'enableRateLimit': True
        } )
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts)
    ohlcv = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # keep last x rows
    ohlcv = ohlcv.tail(showlimit)
    ohlcv['datetime'] = pd.to_datetime(ohlcv['timestamp'], unit='ms')
    return ohlcv