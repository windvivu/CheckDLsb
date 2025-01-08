# File này dùng để build local database từ dữ liệu của sàn Binance, Yfinance...

import functions.buildDB as buildDB
from functions.getDTsqlite import getMaxTimestamp, removeRowDB
import signal
import time
import pandas as pd


source = 'Binance'
since = '1/1/2020'
end = '12/8/2025'       

signal.signal(signal.SIGINT, buildDB._signal_handler)


if __name__ == '__main__':
    
    build_all = False

    print('Build from source:')
    print('\t1. Spot Binance')
    print('\t2. Future Binance (USDT-Margin)')
    print('\t3. Stock Yahoo finance')
    choice = input('Choose source: ')
    if choice.strip() == '1':
        source = 'Binance'
    if choice.strip() == '2':
        source = 'FBinance'
    elif choice.strip() == '3':
        source = 'Yfinance'
    else:
        print('Invalid choice!')
        exit(0)
    
    if source == 'Yfinance':
        pathData = './data/YFstocks/'
        listDt = pd.read_excel('./data/stocklist.xlsx')
    if source == 'FBinance':
        pathData = './data/BNfuture/'
        listDt = pd.read_excel('./data/futurelist.xlsx')
    else:
        pathData = './data/BNspot/'
        listDt = pd.read_excel('./data/spotlist.xlsx')


    listTimeframe = listDt.columns[4:]
    
    if build_all == False:
        listDt = listDt[listDt['build'] == 'x']
    
    for i in range(len(listDt)):
        if buildDB.turnon == False:
            print('Stop loading data!!!')
            break
        
        if source == 'Binance':
            symbol = listDt.iloc[i]['asset'] + '/USDT'
        elif source == 'FBinance':
            symbol = listDt.iloc[i]['asset'] + '/USDT'
        else:
            symbol = listDt.iloc[i]['asset']
        
        count=0
        for timeframe in listTimeframe:
            if buildDB.turnon == False:
                print('Stop loading data!!!')
                break
            if listDt.iloc[i][timeframe] == 'x':
                count += 1
                print(f'{i+1}.{count} - Building', symbol, 'with', timeframe)
                mess = f'{i+1}.{count} - {symbol}.{timeframe} -'
                buildDB.BuilDB(symbol=symbol, since=since, end=end, timeframe=timeframe, mess=mess, pathData=pathData, source=source)
                if buildDB.totalrow > 0:
                    maxTs = getMaxTimestamp(pathData, symbol, timeframe)
                    removeRowDB(pathData, symbol, timeframe, maxTs)
                    print('Lats row was removed from database in order to prevent unstable last candlestick')
                print('Resting in 3 seconds...')
                time.sleep(3)
    
    if buildDB.turnon: exit(0)
    if listDt.shape[0] == 0:
        print('No data to build!')
    else:
        print('All done!') 
    