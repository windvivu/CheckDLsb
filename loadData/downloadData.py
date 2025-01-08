# Lấy dữ liệu ohlcv từ sàn Binance. DÙng để lấy dữ liệu và lưu về local database
# File chính sử dụng hàm này là _sqlite/buildDBfromBN.py

# core load data, return list of list
def _coreLoadBinanceData(symbol, timeframe, since_timestamp, option):
    '''
    option: spot or future
    '''
    import ccxt
    if option == 'spot':
        exchange = ccxt.binance({
            'enableRateLimit': True
        } )
    else: # future
        exchange = ccxt.binanceusdm({
            'enableRateLimit': True
        } )
    data = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp)
    # ccxt trả về list
    return data

def _coreLoadYfinanceData(symbol, timeframe, since_timestamp):
    import yfinance as yf
    import pandas as pd

    if len(str(since_timestamp)) > 10:
        since_timestamp = since_timestamp // 1000

    data = yf.download(symbol, start=since_timestamp, end=None, interval=timeframe)
    data.reset_index(inplace=True)
    if 'Date' in data.columns:
        data = data.rename(columns={'Date': 'Datetime'})
    data = data.rename(columns={'Datetime': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})  
    if data.empty == False:
        data['timestamp'] = data['timestamp'].astype(int) // 10**6 
        data = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # yfinance trả về dataframe nên cần chuyển đổi
    data = data.values.tolist()
    return data

def _convertDataframe(data):
    import pandas as pd
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = df['timestamp'].astype('int64')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def loadDataPeriod(symbol, timeframe, since, end=None, raw_timestap=False, source=''):
    ''' 
    since : yyyy-mm-dd 
    end : yyyy-mm-dd or None
    nếu raw_timestap==True, thì since và end sẽ là dạng số long

    source: 'Yfinance', 'Binance', 'FBinance'
    '''
    import ccxt
    from datetime import datetime
    if source == 'Yfinance':
        print('Loading data from Yfinance...')
    elif source == 'Binance':
        print('Loading data from Spot Binance...')
        exchange = ccxt.binance({
        'enableRateLimit': True
    } )
    elif source == 'FBinance':
        print('Loading data from Future Binance...')
        exchange = ccxt.binanceusdm({
            'enableRateLimit': True
        } )
    else:
        print('Error: source not found')
        return None

    unlimited = False
    if raw_timestap == True:
        since_timestamp = since
    else:
        since_timestamp = exchange.parse8601(since + 'T00:00:00Z')
    if end == None:
        end = datetime.now().strftime('%Y-%m-%d')
        end_timestamp = exchange.parse8601(end + 'T23:59:59Z')
        unlimited = True
    else:
        if raw_timestap == True:
            end_timestamp = end
        else:
            end_timestamp = exchange.parse8601(end + 'T23:59:59Z')
        if end_timestamp < since_timestamp:
            print('Error: End date is earlier than since date')
            return None
    
    # core load data
    if source == 'Yfinance':
        data = _coreLoadYfinanceData(symbol, timeframe, since_timestamp)
    elif source == 'Binance':
        data = _coreLoadBinanceData(symbol, timeframe, since_timestamp, option='spot')
    else:
        data = _coreLoadBinanceData(symbol, timeframe, since_timestamp, option='future')

    lastTimestamp = data[-1][0]
    if lastTimestamp > end_timestamp:
        # slice data to end_timestamp
        while lastTimestamp > end_timestamp:
            data.pop()
            lastTimestamp = data[-1][0]
        df = _convertDataframe(data)
        return df
    
    lastTimestamp +=1
    while lastTimestamp < end_timestamp:
        # core load more data
        if source == 'Yfinance':
            tempdata = _coreLoadYfinanceData(symbol, timeframe, lastTimestamp)
        else:
            tempdata = _coreLoadBinanceData(symbol, timeframe, lastTimestamp)
        if len(tempdata) == 0:
            break
        else:
            data += tempdata
            lastTimestamp = data[-1][0] + 1
    if unlimited == False:
        while lastTimestamp > end_timestamp:
                data.pop()
                lastTimestamp = data[-1][0]
    df = _convertDataframe(data)
    if source == 'Yfinance':
        print('Loading data from Yfinance done!')
    else:
        print('Loading data from Binance done!')
    return df