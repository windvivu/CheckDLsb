import pytz
# tz = pytz.timezone('Asia/Bangkok')
# tz = pytz.timezone('Asia/Oral')
# tz = pytz.timezone('Asia/Ho_Chi_Minh')
tz =pytz.utc   # UTC

def cvtimestamp(stringdate):
    '''
    Áp dụng cho string dạng '2017-09-17T11:59:59Z'
    '''
    import pandas as pd
    return int(pd.Timestamp(stringdate, tz=tz).timestamp()*1000)

def cvstring2timestamp(string_date, toStandardstrT = False):
    '''
    Áp dụng cho string dạng 'dd/MM/yyyy HH:mm:ss'
    '''
    # convert string_date qua dạng 2017-09-17T11:59:59Z
    # change 2 space to 1 space
    string_date = ' '.join(string_date.split())
    # cắt phần ngày và phần giờ
    str = string_date.split(' ')
    # lấy phần ngày
    strd = str[0]
    # khởi tạo phần chứa thời gian
    strt = ''
    if len(str)>1: strt = str[1]

    try:
        if '/' in strd:
            strd = strd.split('/')
            strd = [ d.zfill(2) for d in strd ]
            if len(strd[2]) == 2: strd[2] = '20' + strd[2]
            strd = strd[2]+'-'+strd[1]+'-'+strd[0]
        elif '-' in strd: # _-_-_
            strd = strd.split('-')
            strd = [ d.zfill(2) for d in strd ]
            if len(strd[0]) == 4 and len(strd[1]) == 2 and len(strd[2]) == 2:  # xxxx-xx-xx
                strd = strd[0]+'-'+strd[1]+'-'+strd[2]
            elif len(strd[0]) == 2 and len(strd[1]) == 2 and len(strd[2]) == 4: #xx-xx-xxxx
                strd = strd[2]+'-'+strd[1]+'-'+strd[0]
            elif len(strd[0]) == 2 and len(strd[1]) == 2 and len(strd[2]) == 2: #xx-xx-xx
                strd = '20' + strd[2]+'-'+strd[1]+'-'+strd[0]
            else:
                return 'error'
        else:
            return 'error'
    except:
        return 'error'

    if ':' not in strt:
        strt = 'T00:00:00Z'
    else:
        strt = strt.split(':')
        if len(strt) == 2: 
            strt.append('00')
        
        if len(strt) == 3:
            strt = [ t.zfill(2) for t in strt ]
            strt = 'T'+strt[0]+':'+strt[1]+':'+strt[2]+'Z'
        else:
            strt = 'T00:00:00Z'

    if toStandardstrT: return strd+strt
    return cvtimestamp(strd+strt)















